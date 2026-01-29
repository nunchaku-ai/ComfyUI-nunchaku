import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from huggingface_hub import utils as huggingface_utils
from packaging.version import Version
from safetensors.torch import load_file
from functools import partial

import comfy
import comfy.model_management
import comfy.model_patcher
import folder_paths
import diffusers
import math
import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from comfy.supported_models import Flux, FluxSchnell
from comfy.model_base import BaseModel, convert_tensor, Flux as _Flux, utils

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.models.transformers.transformer_flux import load_quantized_module
from nunchaku.utils import load_state_dict_in_safetensors, get_precision, pad_tensor, check_hardware_compatibility
from nunchaku.lora.flux.compose import compose_lora

from typing import TYPE_CHECKING
from enum import Enum


class ModelType(Enum):
    PIFLOW = 1

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

class BasePolicy(metaclass=ABCMeta):

    @abstractmethod
    def pi(self, x_t, sigma_t):
        """Compute the flow velocity at (x_t, t).

        Args:
            x_t (torch.Tensor): Noisy input at time t.
            sigma_t (torch.Tensor): Noise level at time t.

        Returns:
            torch.Tensor: The computed flow velocity u_t.
        """
        pass

    @abstractmethod
    def detach(self):
        pass

@torch.jit.script
def gmflow_posterior_mean_jit(
        sigma_t_src, sigma_t, x_t_src, x_t,
        gm_means, gm_vars, gm_logweights,
        eps: float, gm_dim: int = -4, channel_dim: int = -3):
    alpha_t_src = 1 - sigma_t_src
    alpha_t = 1 - sigma_t

    sigma_t_src_sq = sigma_t_src.square()
    sigma_t_sq = sigma_t.square()

    # compute gaussian params
    denom = (alpha_t.square() * sigma_t_src_sq - alpha_t_src.square() * sigma_t_sq).clamp(min=eps)  # ζ
    g_mean = (alpha_t * sigma_t_src_sq * x_t - alpha_t_src * sigma_t_sq * x_t_src) / denom  # ν / ζ
    g_var = sigma_t_sq * sigma_t_src_sq / denom

    # gm_mul_iso_gaussian
    g_mean = g_mean.unsqueeze(gm_dim)  # (bs, *, 1, out_channels, h, w)
    g_var = g_var.unsqueeze(gm_dim)  # (bs, *, 1, 1, 1, 1)

    gm_diffs = gm_means - g_mean  # (bs, *, num_gaussians, out_channels, h, w)
    norm_factor = (g_var + gm_vars).clamp(min=eps)

    out_means = (g_var * gm_means + gm_vars * g_mean) / norm_factor
    # (bs, *, num_gaussians, 1, h, w)
    logweights_delta = gm_diffs.square().sum(dim=channel_dim, keepdim=True) * (-0.5 / norm_factor)
    out_weights = (gm_logweights + logweights_delta).softmax(dim=gm_dim)

    out_mean = (out_means * out_weights).sum(dim=gm_dim)

    return out_mean


def gm_temperature(gm, temperature, gm_dim=-4, eps=1e-6):
    gm = gm.copy()
    temperature = max(temperature, eps)
    gm['logweights'] = (gm['logweights'] / temperature).log_softmax(dim=gm_dim)
    if 'logstds' in gm:
        gm['logstds'] = gm['logstds'] + (0.5 * math.log(temperature))
    if 'gm_vars' in gm:
        gm['gm_vars'] = gm['gm_vars'] * temperature
    return gm


class GMFlowPolicy(BasePolicy):
    """GMFlow policy. The number of components K is inferred from the denoising output.

    Args:
        denoising_output (dict): The output of the denoising model, containing:
            means (torch.Tensor): The means of the Gaussian components. Shape (B, K, C, H, W) or (B, K, C, T, H, W).
            logstds (torch.Tensor): The log standard deviations of the Gaussian components. Shape (B, K, 1, 1, 1)
                or (B, K, 1, 1, 1, 1).
            logweights (torch.Tensor): The log weights of the Gaussian components. Shape (B, K, 1, H, W) or
                (B, K, 1, T, H, W).
        x_t_src (torch.Tensor): The initial noisy sample. Shape (B, C, H, W) or (B, C, T, H, W).
        sigma_t_src (torch.Tensor): The initial noise level. Shape (B,).
        checkpointing (bool): Whether to use gradient checkpointing to save memory. Defaults to True.
        eps (float): A small value to avoid numerical issues. Defaults to 1e-4.
    """

    def __init__(
            self,
            denoising_output: Dict[str, torch.Tensor],
            x_t_src: torch.Tensor,
            sigma_t_src: torch.Tensor,
            checkpointing: bool = True,
            eps: float = 1e-4):
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.checkpointing = checkpointing
        self.eps = eps

        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.denoising_output_x_0 = self._u_to_x_0(
            denoising_output, self.x_t_src, self.sigma_t_src)

    @staticmethod
    def _u_to_x_0(denoising_output, x_t, sigma_t):
        x_t = x_t.unsqueeze(1)
        sigma_t = sigma_t.unsqueeze(1)
        means_x_0 = x_t - sigma_t * denoising_output['means']
        gm_vars = (denoising_output['logstds'] * 2).exp() * sigma_t.square()
        return dict(
            means=means_x_0,
            gm_vars=gm_vars,
            logweights=denoising_output['logweights'])

    def pi(self, x_t, sigma_t):
        """Compute the flow velocity at (x_t, t).

        Args:
            x_t (torch.Tensor): Noisy input at time t.
            sigma_t (torch.Tensor): Noise level at time t.

        Returns:
            torch.Tensor: The computed flow velocity u_t.
        """
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        means = self.denoising_output_x_0['means']
        gm_vars = self.denoising_output_x_0['gm_vars']
        logweights = self.denoising_output_x_0['logweights']
        if (sigma_t == self.sigma_t_src).all() and (x_t == self.x_t_src).all():
            x_0 = (logweights.softmax(dim=1) * means).sum(dim=1)
        else:
            if self.checkpointing and torch.is_grad_enabled():
                x_0 = torch.utils.checkpoint.checkpoint(
                    gmflow_posterior_mean_jit,
                    self.sigma_t_src, sigma_t, self.x_t_src, x_t,
                    means,
                    gm_vars,
                    logweights,
                    self.eps, 1, 2,
                    use_reentrant=True)  # use_reentrant=False does not work with jit
            else:
                x_0 = gmflow_posterior_mean_jit(
                    self.sigma_t_src, sigma_t, self.x_t_src, x_t,
                    means,
                    gm_vars,
                    logweights,
                    self.eps, 1, 2)
        u = (x_t - x_0) / sigma_t.clamp(min=self.eps)
        return u

    def copy(self):
        new_policy = GMFlowPolicy.__new__(GMFlowPolicy)
        new_policy.x_t_src = self.x_t_src
        new_policy.ndim = self.ndim
        new_policy.checkpointing = self.checkpointing
        new_policy.eps = self.eps
        new_policy.sigma_t_src = self.sigma_t_src
        new_policy.denoising_output_x_0 = self.denoising_output_x_0.copy()
        return new_policy

    def detach_(self):
        self.denoising_output_x_0 = {k: v.detach() for k, v in self.denoising_output_x_0.items()}
        return self

    def detach(self):
        new_policy = self.copy()
        return new_policy.detach_()

    def dropout_(self, p):
        if p <= 0 or p >= 1:
            return self
        logweights = self.denoising_output_x_0['logweights']
        dropout_mask = torch.rand(
            (*logweights.shape[:2], *((self.ndim - 1) * [1])), device=logweights.device) < p
        is_all_dropout = dropout_mask.all(dim=1, keepdim=True)
        dropout_mask &= ~is_all_dropout
        self.denoising_output_x_0['logweights'] = logweights.masked_fill(
            dropout_mask, float('-inf'))
        return self

    def dropout(self, p):
        new_policy = self.copy()
        return new_policy.dropout_(p)

    def temperature_(self, temp):
        if temp >= 1.0:
            return self
        self.denoising_output_x_0 = gm_temperature(
            self.denoising_output_x_0, temp, gm_dim=1, eps=self.eps)
        return self

    def temperature(self, temp):
        new_policy = self.copy()
        return new_policy.temperature_(temp)

class ModelSamplingPiFlow(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(
            shift=sampling_settings.get("shift", 3.2),
            multiplier=sampling_settings.get("multiplier", 1.0))

    def set_parameters(self, shift=3.2, multiplier=1.0):
        self.shift = shift
        self.multiplier = multiplier

    def timestep(self, sigma):
        return sigma * self.multiplier

    def warp_t(self, t):
        shift = self.shift
        return shift * t / (1 + (shift - 1) * t)

    def unwarp_t(self, t):
        shift = self.shift
        return t / (shift + (1 - shift) * t)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return self.warp_t(1.0 - percent)

def model_sampling(model_config, model_type):
    if model_type == ModelType.PIFLOW:
        c = comfy.model_sampling.CONST
        s = ModelSamplingPiFlow
    else:
        raise ValueError("Unsupported model type {}".format(model_type))

    class ModelSampling(s, c):
        pass

    return ModelSampling(model_config)

class DXPolicy(BasePolicy):
    """DX policy. The number of grid points N is inferred from the denoising output.

    Note: segment_size and shift are intrinsic parameters of the DX policy. For elastic inference (i.e., changing
    the number of function evaluations or noise schedule at test time), these parameters should be kept unchanged.

    Args:
        denoising_output (torch.Tensor): The output of the denoising model. Shape (B, N, C, H, W) or (B, N, C, T, H, W).
        x_t_src (torch.Tensor): The initial noisy sample. Shape (B, C, H, W) or (B, C, T, H, W).
        sigma_t_src (torch.Tensor): The initial noise level. Shape (B,).
        segment_size (float): The size of each DX policy time segment. Defaults to 1.0.
        shift (float): The shift parameter for the DX policy noise schedule. Defaults to 1.0.
        mode (str): Either 'grid' or 'polynomial' mode for calculating x_0. Defaults to 'grid'.
        eps (float): A small value to avoid numerical issues. Defaults to 1e-4.
    """

    def __init__(
            self,
            denoising_output: torch.Tensor,
            x_t_src: torch.Tensor,
            sigma_t_src: torch.Tensor,
            segment_size: float = 1.0,
            shift: float = 1.0,
            mode: str = 'grid',
            eps: float = 1e-4):
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.shift = shift
        self.eps = eps

        assert mode in ['grid', 'polynomial']
        self.mode = mode

        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.raw_t_src = self._unwarp_t(self.sigma_t_src)
        self.raw_t_dst = (self.raw_t_src - segment_size).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)

        self.denoising_output_x_0 = self._u_to_x_0(
            denoising_output, self.x_t_src, self.sigma_t_src)

    def _unwarp_t(self, sigma_t):
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    @staticmethod
    def _u_to_x_0(denoising_output, x_t, sigma_t):
        x_0 = x_t.unsqueeze(1) - sigma_t.unsqueeze(1) * denoising_output
        return x_0

    @staticmethod
    def _interpolate(x, t):
        """
        Args:
            x (torch.Tensor): (B, N, *)
            t (torch.Tensor): (B, *) in [0, 1]

        Returns:
            torch.Tensor: (B, *)
        """
        n = x.size(1)
        if n < 2:
            return x.squeeze(1)
        t = t.clamp(min=0, max=1) * (n - 1)
        t0 = t.floor().to(torch.long).clamp(min=0, max=n - 2)
        t1 = t0 + 1
        t0t1 = torch.stack([t0, t1], dim=1)  # (B, 2, *)
        x0x1 = torch.gather(x, dim=1, index=t0t1.expand(-1, -1, *x.shape[2:]))
        x_interp = (t1 - t) * x0x1[:, 0] + (t - t0) * x0x1[:, 1]
        return x_interp

    def pi(self, x_t, sigma_t):
        """Compute the flow velocity at (x_t, t).

        Args:
            x_t (torch.Tensor): Noisy input at time t.
            sigma_t (torch.Tensor): Noise level at time t.

        Returns:
            torch.Tensor: The computed flow velocity u_t.
        """
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        raw_t = self._unwarp_t(sigma_t)
        if self.mode == 'grid':
            x_0 = self._interpolate(
                self.denoising_output_x_0, (raw_t - self.raw_t_dst) / self.segment_size)
        elif self.mode == 'polynomial':
            p_order = self.denoising_output_x_0.size(1)
            diff_t = self.raw_t_src - raw_t  # (B, 1, 1, 1)
            basis = torch.stack(
                [diff_t ** i for i in range(p_order)], dim=1)  # (B, N, 1, 1, 1)
            x_0 = torch.sum(basis * self.denoising_output_x_0, dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        u = (x_t - x_0) / sigma_t.clamp(min=self.eps)
        return u

    def copy(self):
        new_policy = DXPolicy.__new__(DXPolicy)
        new_policy.x_t_src = self.x_t_src
        new_policy.ndim = self.ndim
        new_policy.shift = self.shift
        new_policy.eps = self.eps
        new_policy.mode = self.mode
        new_policy.sigma_t_src = self.sigma_t_src
        new_policy.raw_t_src = self.raw_t_src
        new_policy.raw_t_dst = self.raw_t_dst
        new_policy.segment_size = self.segment_size
        new_policy.denoising_output_x_0 = self.denoising_output_x_0
        return new_policy

    def detach_(self):
        self.denoising_output_x_0 = self.denoising_output_x_0.detach()
        return self

    def detach(self):
        new_policy = self.copy()
        return new_policy.detach_()

POLICY_CLASSES = dict(
    DX=DXPolicy,
    GMFlow=GMFlowPolicy
)

class BasePiFlow(BaseModel):

    def __init__(self, model_config, diffusion_model, model_type=ModelType.PIFLOW, device=None):
        super(BaseModel, self).__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        self.current_patcher: 'ModelPatcher' = None

        if not unet_config.get("disable_unet_model_creation", False):
            if model_config.custom_operations is None:
                fp8 = model_config.optimizations.get("fp8", False)
                kwargs = dict(fp8_optimizations=fp8, scaled_fp8=model_config.scaled_fp8,)
                if model_config and hasattr(model_config, 'layer_quant_config') and model_config.layer_quant_config:
                    kwargs.update(model_config=model_config)
                operations = comfy.ops.pick_operations(
                    unet_config.get("dtype", None), self.manual_cast_dtype, **kwargs)
            else:
                operations = model_config.custom_operations
            self.diffusion_model = diffusion_model(**unet_config, device=device, operations=operations)
            self.diffusion_model.eval()
            if comfy.model_management.force_channels_last():
                self.diffusion_model.to(memory_format=torch.channels_last)
                logging.debug("using channels last mode for diffusion model")
            logging.info("model weight dtype {}, manual cast: {}".format(self.get_dtype(), self.manual_cast_dtype))
        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0

        self.concat_keys = ()
        logging.info("model_type {}".format(model_type.name))
        logging.debug("adm {}".format(self.adm_channels))
        self.memory_usage_factor = model_config.memory_usage_factor
        self.memory_usage_factor_conds = ()
        self.memory_usage_shape_process = {}

        policy_config = model_config.policy_config.copy()
        policy_type = policy_config.pop("type")
        self.policy_class = partial(POLICY_CLASSES[policy_type], **policy_config)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        if c_concat is not None:
            xc = torch.cat([xc] + [comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)], dim=1)

        context = c_crossattn
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        xc = xc.to(dtype)
        device = xc.device
        t = self.model_sampling.timestep(t).float()
        if context is not None:
            context = comfy.model_management.cast_to_device(context, device, dtype)

        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]

            if hasattr(extra, "dtype"):
                extra = convert_tensor(extra, dtype, device)
            elif isinstance(extra, list):
                ex = []
                for ext in extra:
                    ex.append(convert_tensor(ext, dtype, device))
                extra = ex
            extra_conds[o] = extra

        t = self.process_timestep(t, x=x, **extra_conds)
        assert "latent_shapes" not in extra_conds, \
            "`pack_latents` and `unpack_latents` are currently not supported in PiFlow models."

        model_output = self.diffusion_model(xc, t, context=context, control=control,
                                            transformer_options=transformer_options, **extra_conds)
        if isinstance(model_output, dict):
            model_output = {k: v.float() for k, v in model_output.items()}
        else:
            model_output = model_output.float()
        return self.policy_class(model_output, x, sigma)




class NunchakuFluxTransformer2dModelPiFlow(NunchakuFluxTransformer2dModel):
    """
    NunchakuFluxTransformer2dModel with added PiFlow functionality

    Parameters
    ----------
    patch_size : int, optional
        Patch size for input images (default: 2).
    in_channels : int, optional
        Number of input channels (default: 64).
    out_channels : int or None, optional
        Number of output channels (default: None).
    num_layers : int, optional
        Number of transformer layers (default: 19).
    num_single_layers : int, optional
        Number of single transformer layers (default: 38).
    attention_head_dim : int, optional
        Dimension of each attention head (default: 128).
    num_attention_heads : int, optional
        Number of attention heads (default: 24).
    joint_attention_dim : int, optional
        Joint attention dimension (default: 4096).
    pooled_projection_dim : int, optional
        Pooled projection dimension (default: 768).
    guidance_embeds : bool, optional
        Whether to use guidance embeddings (default: False).
    axes_dims_rope : tuple[int], optional
        Axes dimensions for rotary embeddings (default: (16, 56, 56)).

    """
    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int] = (16, 56, 56),
        num_gaussians: int = 8,
        constant_logstd: int = None,
        logstd_inner_dim : int = 1024,
        gm_num_logstd_layers : int = 2,
    ):

        super(NunchakuFluxTransformer2dModelPiFlow, self).__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

        self.num_gaussians = num_gaussians
        self.constant_logstd = constant_logstd
        self.logstd_inner_dim = logstd_inner_dim
        self.gm_num_logstd_layers = gm_num_logstd_layers
        self.patch_size = patch_size

        self.proj_out = None

        self.proj_out_means = nn.Linear(
            self.inner_dim, self.num_gaussians * self.out_channels,
            bias=True)

        self.proj_out_logweights = nn.Linear(
            self.inner_dim, self.num_gaussians * self.patch_size * self.patch_size,
            bias=True)

        if self.constant_logstd is None:
            assert gm_num_logstd_layers >= 1
            in_dim = self.inner_dim
            logstd_layers = []
            for _ in range(gm_num_logstd_layers - 1):
                logstd_layers.extend([
                    nn.SiLU(),
                    nn.Linear(in_dim, logstd_inner_dim, bias=True)])
                in_dim = logstd_inner_dim
            self.proj_out_logstds = nn.Sequential(
                *logstd_layers,
                nn.SiLU(),
                nn.Linear(in_dim, 1, bias=True))

    @classmethod
    def _build_model(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs
    ) -> tuple[nn.Module, dict[str, torch.Tensor], dict[str, str]]:
        """
        Build a transformer model from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file.
        **kwargs
            Additional keyword arguments (e.g., ``torch_dtype``).

        Returns
        -------
        tuple
            (transformer, state_dict, metadata)
        """
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        state_dict, metadata = load_state_dict_in_safetensors(pretrained_model_name_or_path, return_metadata=True)

        config = json.loads(metadata["config"])

        if "patch_size" in config:
            config["patch_size"] = 2

        with torch.device("meta"):
            transformer = cls.from_config(config).to(kwargs.get("torch_dtype", torch.bfloat16))

        return transformer, state_dict, metadata

    @classmethod
    @huggingface_utils.validate_hf_hub_args
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str | os.PathLike[str],
                        adapter_model_name_or_path: str | os.PathLike[str],
                        adapter_strength: float, **kwargs):
        """
        Loads a Nunchaku FLUX transformer model *and* relevant PiFlow adapter from pretrained weights.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the FLUX model directory or HuggingFace repo.
        adapter_model_name_or_path : str or os.PathLike
            Path to the PiFlow adapter directory or HuggingFace repo.
        adapter_strength : float
            PiFlow adapter strength.
        **kwargs
            Additional keyword arguments for device, offload, torch_dtype, precision, etc.

        Returns
        -------
        NunchakuFluxTransformer2dModelPiFlow or (NunchakuFluxTransformer2dModelPiFlow, dict)
            The loaded model, and optionally metadata if `return_metadata=True`.
        """
        device = kwargs.get("device", "cuda")
        if isinstance(device, str):
            device = torch.device(device)
        offload = kwargs.get("offload", False)
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        precision = get_precision(kwargs.get("precision", "auto"), device, pretrained_model_name_or_path)
        metadata = None

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        if pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ):
            transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
            quantized_part_sd = {}
            unquantized_part_sd = {}
            for k, v in model_state_dict.items():
                if k.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                    quantized_part_sd[k] = v
                else:
                    unquantized_part_sd[k] = v
            precision = get_precision(device=device)
            quantization_config = json.loads(metadata["quantization_config"])
            check_hardware_compatibility(quantization_config, device)
        else:
            transformer, unquantized_part_path, transformer_block_path = cls._build_model_legacy(
                pretrained_model_name_or_path, **kwargs
            )

            # get the default LoRA branch and all the vectors
            quantized_part_sd = load_file(transformer_block_path)
            unquantized_part_sd = load_file(unquantized_part_path)
        new_quantized_part_sd = {}
        for k, v in quantized_part_sd.items():
            if v.ndim == 1:
                new_quantized_part_sd[k] = v
            elif "qweight" in k:
                # only the shape information of this tensor is needed
                new_quantized_part_sd[k] = v.to("meta")

                # if the tensor has qweight, but does not have low-rank branch, we need to add some artificial tensors
                for t in ["lora_up", "lora_down"]:
                    new_k = k.replace(".qweight", f".{t}")
                    if new_k not in quantized_part_sd:
                        oc, ic = v.shape
                        ic = ic * 2  # v is packed into INT8, so we need to double the size
                        new_quantized_part_sd[k.replace(".qweight", f".{t}")] = torch.zeros(
                            (0, ic) if t == "lora_down" else (oc, 0), device=v.device, dtype=torch.bfloat16
                        )

            elif "lora" in k:
                new_quantized_part_sd[k] = v
        transformer._quantized_part_sd = new_quantized_part_sd
        m = load_quantized_module(
            quantized_part_sd,
            device=device,
            use_fp4=precision == "fp4",
            offload=offload,
            bf16=torch_dtype == torch.bfloat16,
        )
        transformer.inject_quantized_module(m, device)
        transformer.to_empty(device=device)

        #load the PiFlow adapter
        adapter_state_dict, adapter_metadata = load_state_dict_in_safetensors(adapter_model_name_or_path, return_metadata=True)

        #PiFlow-specific model weights get added to unquantized_part_sd, the rest are treated as a LoRA

        adapter_state_dict_lora = {}
        
        for key in list(adapter_state_dict.keys()):
            value = adapter_state_dict.pop(key).to(torch.bfloat16) #nunchaku svdq weights are in bf16, while adapter weights are in f16, need to convert
            if "lora" in key:
                adapter_state_dict_lora[key] = value
            else:
                unquantized_part_sd[key] = value


        transformer.load_state_dict(unquantized_part_sd, strict=False)
        transformer._unquantized_part_sd = unquantized_part_sd

        transformer.update_lora_params(compose_lora([(adapter_state_dict_lora, adapter_strength)]))

        if kwargs.get("return_metadata", False):
            return transformer, metadata
        else:
            return transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        Forward pass for the Nunchaku FLUX transformer model.

        This method is compatible with the Diffusers pipeline and supports LoRA,
        rotary embeddings, and ControlNet.

        Parameters
        ----------
        hidden_states : torch.FloatTensor
            Input hidden states of shape (batch_size, channel, height, width).
        encoder_hidden_states : torch.FloatTensor, optional
            Conditional embeddings (e.g., prompt embeddings) of shape (batch_size, sequence_len, embed_dims).
        pooled_projections : torch.FloatTensor, optional
            Embeddings projected from the input conditions.
        timestep : torch.LongTensor, optional
            Denoising step.
        img_ids : torch.Tensor, optional
            Image token indices.
        txt_ids : torch.Tensor, optional
            Text token indices.
        guidance : torch.Tensor, optional
            Guidance tensor for classifier-free guidance.
        joint_attention_kwargs : dict, optional
            Additional kwargs for joint attention.
        controlnet_block_samples : list[torch.Tensor], optional
            ControlNet block samples.
        controlnet_single_block_samples : list[torch.Tensor], optional
            ControlNet single block samples.
        return_dict : bool, optional
            Whether to return a Transformer2DModelOutput (default: True).
        controlnet_blocks_repeat : bool, optional
            Whether to repeat ControlNet blocks (default: False).

        Returns
        -------
        torch.FloatTensor or Transformer2DModelOutput
            Output tensor or output object containing the sample.
        """
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        nunchaku_block = self.transformer_blocks[0]
        encoder_hidden_states, hidden_states = nunchaku_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )
        hidden_states = self.norm_out(hidden_states, temb)

        #pi-flow skips the original proj_out layer for its own layers
        output = hidden_states
        #output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

class GMFlux(BasePiFlow, _Flux):
    def __init__(self, model_config, device=None):
        super().__init__(model_config, NunchakuFluxTransformer2dModelPiFlow, device=device)
        self.memory_usage_factor_conds = ("ref_latents",)

    def concat_cond(self, **kwargs):
        try:
            # Handle Flux control loras dynamically changing the img_in weight.
            num_channels = self.diffusion_model.img_in.weight.shape[1] // (self.diffusion_model.patch_size * self.diffusion_model.patch_size)
        except:
            # Some cases like tensorrt might not have the weights accessible
            num_channels = self.model_config.unet_config["in_channels"]

        out_channels = self.model_config.unet_config["out_channels"]

        if num_channels <= out_channels:
            return None

        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros_like(noise)

        image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        image = utils.resize_to_batch_size(image, noise.shape[0])
        image = self.process_latent_in(image)
        if num_channels <= out_channels * 2:
            return image

        # inpaint model
        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.ones_like(noise)[:, :1]

        mask = torch.mean(mask, dim=1, keepdim=True)
        mask = utils.common_upscale(mask.to(device), noise.shape[-1] * 8, noise.shape[-2] * 8, "bilinear", "center")
        mask = mask.view(mask.shape[0], mask.shape[2] // 8, 8, mask.shape[3] // 8, 8).permute(0, 2, 4, 1, 3).reshape(mask.shape[0], -1, mask.shape[2] // 8, mask.shape[3] // 8)
        mask = utils.resize_to_batch_size(mask, noise.shape[0])
        return torch.cat((image, mask), dim=1)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        # upscale the attention mask, since now we
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            shape = kwargs["noise"].shape
            mask_ref_size = kwargs["attention_mask_img_shape"]
            # the model will pad to the patch size, and then divide
            # essentially dividing and rounding up
            (h_tok, w_tok) = (math.ceil(shape[2] / self.diffusion_model.patch_size), math.ceil(shape[3] / self.diffusion_model.patch_size))
            attention_mask = utils.upscale_dit_mask(attention_mask, mask_ref_size, (h_tok, w_tok))
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)

        guidance = kwargs.get("guidance", 3.5)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))

        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            latents = []
            for lat in ref_latents:
                latents.append(self.process_latent_in(lat))
            out['ref_latents'] = comfy.conds.CONDList(latents)

            ref_latents_method = kwargs.get("reference_latents_method", None)
            if ref_latents_method is not None:
                out['ref_latents_method'] = comfy.conds.CONDConstant(ref_latents_method)
        return out

    def extra_conds_shapes(self, **kwargs):
        out = {}
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            out['ref_latents'] = list([1, 16, sum(map(lambda a: math.prod(a.size()), ref_latents)) // 16])
        return out


        