"""
This module provides a wrapper for the :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel`,
enabling integration with ComfyUI forward,
LoRA composition, and advanced caching strategies.
"""

from typing import Callable, Tuple

import torch
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.model_patcher import ModelPatcher
from einops import rearrange, repeat
from torch import nn

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.fbcache import cache_context, create_cache_context
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.utils import load_state_dict_in_safetensors


class ComfyFluxWrapper(nn.Module):
    """
    Wrapper for :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel`
    to support ComfyUI workflows, LoRA composition, and caching.

    Parameters
    ----------
    model : :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel`
        The underlying Nunchaku model to wrap.
    config : dict
        Model configuration dictionary.
    pulid_pipeline : :class:`~nunchaku.pipeline.pipeline_flux_pulid.PuLIDPipeline`, optional
        Optional pipeline for Pulid integration.
    customized_forward : Callable, optional
        Optional custom forward function.
    forward_kwargs : dict, optional
        Additional keyword arguments for the forward pass.
    ctx_for_copy:
        A dict that holds initialization context for later duplication of this ComfyFluxWrapper object.

    Attributes
    ----------
    model : :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel`
        The wrapped model.
    dtype : torch.dtype
        Data type of the model parameters.
    config : dict
        Model configuration.
    loras : list
        List of LoRA metadata for composition.
    pulid_pipeline : :class:`~nunchaku.pipeline.pipeline_flux_pulid.PuLIDPipeline` or None
        Pulid pipeline if provided.
    customized_forward : Callable or None
        Custom forward function if provided.
    forward_kwargs : dict
        Additional arguments for the forward pass.
    ctx_for_copy:
        A dict that holds initialization context for later duplication of this ComfyFluxWrapper object.
    """

    def __init__(
        self,
        model: NunchakuFluxTransformer2dModel,
        config,
        pulid_pipeline=None,
        customized_forward: Callable = None,
        forward_kwargs: dict | None = {},
        ctx_for_copy: dict = {},
    ):
        super(ComfyFluxWrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        self.loras = []

        self.pulid_pipeline = pulid_pipeline
        self.customized_forward = customized_forward
        self.forward_kwargs = {} if forward_kwargs is None else forward_kwargs

        self.ctx_for_copy = ctx_for_copy.copy()

        self._prev_timestep = None  # for first-block cache
        self._cache_context = None

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        """
        Preprocess an input image tensor for the model.

        Pads and rearranges the image into patches and generates corresponding image IDs.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, channels, height, width).
        index : int, optional
            Index for image ID encoding.
        h_offset : int, optional
            Height offset for patch IDs.
        w_offset : int, optional
            Width offset for patch IDs.

        Returns
        -------
        img : torch.Tensor
            Rearranged image tensor of shape (batch, num_patches, patch_dim).
        img_ids : torch.Tensor
            Image ID tensor of shape (batch, num_patches, 3).
        """
        bs, c, h, w = x.shape
        patch_size = self.config.get("patch_size", 2)
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size

        h_offset = (h_offset + (patch_size // 2)) // patch_size
        w_offset = (w_offset + (patch_size // 2)) // patch_size

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def forward(
        self,
        x,
        timestep,
        context,
        y,
        guidance,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        """
        Forward pass for the wrapped model.

        Handles LoRA composition, caching, PuLID integration, and reference latents.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor.
        timestep : float or torch.Tensor
            Diffusion timestep.
        context : torch.Tensor
            Context tensor (e.g., text embeddings).
        y : torch.Tensor
            Pooled projections or additional conditioning.
        guidance : torch.Tensor
            Guidance embedding or value.
        control : dict, optional
            ControlNet input and output samples.
        transformer_options : dict, optional
            Additional transformer options.
        **kwargs
            Additional keyword arguments, e.g., 'ref_latents'.

        Returns
        -------
        out : torch.Tensor
            Output tensor of the same spatial size as the input.
        """
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            assert isinstance(timestep, float)
            timestep_float = timestep

        model = self.model
        assert isinstance(model, NunchakuFluxTransformer2dModel)

        bs, c, h_orig, w_orig = x.shape
        patch_size = self.config.get("patch_size", 2)
        h_len = (h_orig + (patch_size // 2)) // patch_size
        w_len = (w_orig + (patch_size // 2)) // patch_size

        img, img_ids = self.process_img(x)
        img_tokens = img.shape[1]

        ref_latents = kwargs.get("ref_latents")
        if ref_latents is not None:
            h = 0
            w = 0
            for ref in ref_latents:
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h

                kontext, kontext_ids = self.process_img(ref, index=1, h_offset=h_offset, w_offset=w_offset)
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

        # load and compose LoRA
        if self.loras != model.comfy_lora_meta_list:
            lora_to_be_composed = []
            for _ in range(max(0, len(model.comfy_lora_meta_list) - len(self.loras))):
                model.comfy_lora_meta_list.pop()
                model.comfy_lora_sd_list.pop()
            for i in range(len(self.loras)):
                meta = self.loras[i]
                if i >= len(model.comfy_lora_meta_list):
                    sd = load_state_dict_in_safetensors(meta[0])
                    model.comfy_lora_meta_list.append(meta)
                    model.comfy_lora_sd_list.append(sd)
                elif model.comfy_lora_meta_list[i] != meta:
                    if meta[0] != model.comfy_lora_meta_list[i][0]:
                        sd = load_state_dict_in_safetensors(meta[0])
                        model.comfy_lora_sd_list[i] = sd
                    model.comfy_lora_meta_list[i] = meta
                lora_to_be_composed.append(({k: v for k, v in model.comfy_lora_sd_list[i].items()}, meta[1]))

            composed_lora = compose_lora(lora_to_be_composed)

            if len(composed_lora) == 0:
                model.reset_lora()
            else:
                if "x_embedder.lora_A.weight" in composed_lora:
                    new_in_channels = composed_lora["x_embedder.lora_A.weight"].shape[1]
                    current_in_channels = model.x_embedder.in_features
                    if new_in_channels < current_in_channels:
                        model.reset_x_embedder()
                model.update_lora_params(composed_lora)

        controlnet_block_samples = None if control is None else [y.to(x.dtype) for y in control["input"]]
        controlnet_single_block_samples = None if control is None else [y.to(x.dtype) for y in control["output"]]

        if self.pulid_pipeline is not None:
            self.model.transformer_blocks[0].pulid_ca = self.pulid_pipeline.pulid_ca

        if getattr(model, "residual_diff_threshold_multi", 0) != 0 or getattr(model, "_is_cached", False):
            # A more robust caching strategy
            cache_invalid = False

            # Check if timestamps have changed or are out of valid range
            if self._prev_timestep is None:
                cache_invalid = True
            elif self._prev_timestep < timestep_float + 1e-5:  # allow a small tolerance to reuse the cache
                cache_invalid = True

            if cache_invalid:
                self._cache_context = create_cache_context()

            # Update the previous timestamp
            self._prev_timestep = timestep_float
            with cache_context(self._cache_context):
                if self.customized_forward is None:
                    out = model(
                        hidden_states=img,
                        encoder_hidden_states=context,
                        pooled_projections=y,
                        timestep=timestep,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance if self.config["guidance_embed"] else None,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                    ).sample
                else:
                    out = self.customized_forward(
                        model,
                        hidden_states=img,
                        encoder_hidden_states=context,
                        pooled_projections=y,
                        timestep=timestep,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance if self.config["guidance_embed"] else None,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        **self.forward_kwargs,
                    ).sample
        else:
            if self.customized_forward is None:
                out = model(
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                ).sample
            else:
                out = self.customized_forward(
                    model,
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    **self.forward_kwargs,
                ).sample
        if self.pulid_pipeline is not None:
            self.model.transformer_blocks[0].pulid_ca = None

        out = out[:, :img_tokens]
        out = rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_len,
            w=w_len,
            ph=patch_size,
            pw=patch_size,
        )
        out = out[:, :, :h_orig, :w_orig]

        self._prev_timestep = timestep_float
        return out


def copy_with_ctx(model_wrapper: ComfyFluxWrapper) -> Tuple[ComfyFluxWrapper, ModelPatcher]:
    """
    Duplicates a ComfyFluxWrapper object with it's initialization context such as comfy_config, model_config, device and device_id.

    Also create a ModelPatcher object that holds the model_base object created by the model_config.

    Parameters
    ----------
    model_wrapper : ComfyFluxWrapper
        the object to be copied.

    Returns
    -------
    tuple[ComfyFluxWrapper, ModelPatcher]
        the copied ComfyFluxWrapper object and the created ModelPatcher object.
    """
    ctx_for_copy = model_wrapper.ctx_for_copy
    ret_model_wrapper: ComfyFluxWrapper = ComfyFluxWrapper(
        model_wrapper.model,
        config=ctx_for_copy["comfy_config"]["model_config"],
        ctx_for_copy={
            "comfy_config": ctx_for_copy["comfy_config"],
            "model_config": ctx_for_copy["model_config"],
            "device": ctx_for_copy["device"],
            "device_id": ctx_for_copy["device_id"],
        },
    )
    model_base = ctx_for_copy["model_config"].get_model({})
    model_base.diffusion_model = ret_model_wrapper
    ret_model = ModelPatcher(model_base, ctx_for_copy["device"], ctx_for_copy["device_id"])
    return ret_model_wrapper, ret_model
