"""
This module provides a wrapper for the :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel`,
enabling integration with ComfyUI forward,
LoRA composition, and advanced caching strategies.
"""

from typing import Callable

import torch
import comfy
from comfy.ldm.common_dit import pad_to_patch_size
from comfy import latent_formats, supported_models_base
from einops import rearrange, repeat
from torch import nn

from ..models.transformer_flux_piflow import NunchakuFluxTransformer2dModelPiFlow, GMFlux as _GMFlux
from nunchaku.caching.fbcache import cache_context, create_cache_context
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.utils import load_state_dict_in_safetensors
from .flux import ComfyFluxWrapper

class GMFlux(supported_models_base.BASE):
    unet_config = {
        "image_model": "gm_flux",
        "guidance_embed": True,
    }

    policy_config = {
        "type": "GMFlow",
    }

    sampling_settings = {'multiplier': 1.0}

    unet_extra_config = {}
    latent_format = latent_formats.Flux

    memory_usage_factor = 3.1

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = _GMFlux(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.flux.FluxTokenizer, comfy.text_encoders.flux.flux_clip(**t5_detect))


class ComfyFluxPiFlowWrapper(ComfyFluxWrapper):
    """
    Wrapper for :class:`~NunchakuFluxTransformer2dModelPiFlow`
    to support ComfyUI workflows, LoRA composition, and caching.

    Parameters
    ----------
    model : :class:`~NunchakuFluxTransformer2dModelPiFlow`
        The underlying Nunchaku model to wrap.
    config : dict
        Model configuration dictionary.
    pulid_pipeline : :class:`~nunchaku.pipeline.pipeline_flux_pulid.PuLIDPipeline`, optional
        Optional pipeline for Pulid integration.
    customized_forward : Callable, optional
        Optional custom forward function.
    forward_kwargs : dict, optional
        Additional keyword arguments for the forward pass.

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
    """

    def __init__(
        self,
        model: NunchakuFluxTransformer2dModelPiFlow,
        config,
        pulid_pipeline=None,
        customized_forward: Callable = None,
        forward_kwargs: dict | None = {},
    ):
        super(ComfyFluxPiFlowWrapper, self).__init__(model, config)

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
        assert isinstance(model, NunchakuFluxTransformer2dModelPiFlow)

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

        self._prev_timestep = timestep_float

        #alter forward to work with PiFlow Sampler

        #to preserve original logic temb is redundantly calculated
        temb = (
            model.time_text_embed(timestep, y)
            if guidance is None
            else model.time_text_embed(timestep, guidance, y)
        )
    

        bs = out.size(0)
        k = model.num_gaussians
        c = model.out_channels
        patch_size = 2
        out_means = model.proj_out_means(out).reshape(
            bs, h_len, w_len, k, c // (patch_size * patch_size), patch_size, patch_size
        ).permute(
            0, 3, 4, 1, 5, 2, 6
        ).reshape(
            bs, k, c // (patch_size * patch_size), h_len * patch_size, w_len * patch_size
        )[..., :h_orig, :w_orig]
        out_logweights = model.proj_out_logweights(out).reshape(
            bs, h_len, w_len, k, 1, patch_size, patch_size
        ).permute(
            0, 3, 4, 1, 5, 2, 6
        ).reshape(
            bs, k, 1, h_len * patch_size, w_len * patch_size
        )[..., :x.shape[-2], :x.shape[-1]].log_softmax(dim=1)
        if model.constant_logstd is None:
            out_logstds = model.proj_out_logstds(temb.detach()).reshape(bs, 1, 1, 1, 1)
        else:
            out_logstds = out.new_full((bs, 1, 1, 1, 1), float(model.constant_logstd))

        return dict(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)
