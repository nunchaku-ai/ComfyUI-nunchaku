"""
Nunchaku Qwen-Image model base.

This module provides a wrapper for ComfyUI's Qwen-Image model base.
"""

import torch
from comfy.model_base import ModelType, QwenImage

from nunchaku.models.linear import SVDQW4A4Linear

from ..models.qwenimage import NunchakuQwenImageTransformer2DModel


class NunchakuQwenImage(QwenImage):
    """
    Wrapper for the Nunchaku Qwen-Image model.

    Parameters
    ----------
    model_config : object
        Model configuration object.
    model_type : ModelType, optional
        Type of the model (default is ModelType.FLUX).
    device : torch.device or str, optional
        Device to load the model onto.
    """

    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        """
        Initialize the NunchakuQwenImage model.

        Parameters
        ----------
        model_config : object
            Model configuration object.
        model_type : ModelType, optional
            Type of the model (default is ModelType.FLUX).
        device : torch.device or str, optional
            Device to load the model onto.
        """
        super(QwenImage, self).__init__(
            model_config, model_type, device=device, unet_model=NunchakuQwenImageTransformer2DModel
        )
        self.memory_usage_factor_conds = ("ref_latents",)

    def load_model_weights(self, sd: dict[str, torch.Tensor], unet_prefix: str = ""):
        """
        Load model weights into the diffusion model.

        Parameters
        ----------
        sd : dict of str to torch.Tensor
            State dictionary containing model weights.
        unet_prefix : str, optional
            Prefix for UNet weights (default is "").

        Raises
        ------
        ValueError
            If a required key is missing from the state dictionary.
        """
        diffusion_model = self.diffusion_model
        state_dict = diffusion_model.state_dict()
        for k in state_dict.keys():
            if k not in sd:
                if ".wcscales" not in k:
                    raise ValueError(f"Key {k} not found in state_dict")
                sd[k] = torch.ones_like(state_dict[k])
        for n, m in diffusion_model.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                if m.wtscale is not None:
                    m.wtscale = sd.pop(f"{n}.wtscale", 1.0)
        
        # CRITICAL FIX: Fill _quantized_part_sd for LoRA support (following Flux approach)
        # Store proj_down, proj_up, and qweight for LoRA merging
        new_quantized_part_sd = {}
        for k, v in sd.items():
            if v.ndim == 1:
                # Store all 1D tensors (biases, scales)
                new_quantized_part_sd[k] = v
            elif "qweight" in k:
                # Store qweight shape info
                new_quantized_part_sd[k] = v.to("meta")
            elif "proj_down" in k or "proj_up" in k:
                # Store REAL low-rank branches for LoRA merging and restoration
                # Unlike Flux (which uses empty tensors), Qwen Image needs real weights
                # for proper cleanup when removing LoRAs
                new_quantized_part_sd[k] = v
            elif "lora" in k:
                # Store all lora-related keys (same as Flux implementation)
                # This ensures reset_lora() can properly restore original weights
                new_quantized_part_sd[k] = v
        
        diffusion_model._quantized_part_sd = new_quantized_part_sd
        
        # CRITICAL FIX: Initialize clean LoRA state to prevent pollution
        # Clear any existing LoRA state and reset all LoRA strengths to 0
        diffusion_model.comfy_lora_meta_list = []
        diffusion_model.comfy_lora_sd_list = []
        diffusion_model._lora_state_cache = {}
        
        # Reset all LoRA strengths to 0 for a clean start
        self._reset_all_lora_strength_clean(diffusion_model)
        
        diffusion_model.load_state_dict(sd, strict=True)
    
    def _reset_all_lora_strength_clean(self, diffusion_model):
        """
        Reset LoRA strength to 0 for all SVDQW4A4Linear layers in the diffusion model.
        This ensures a clean start without any residual LoRA effects.
        """
        from nunchaku.models.linear import SVDQW4A4Linear
        
        for name, module in diffusion_model.named_modules():
            if isinstance(module, SVDQW4A4Linear):
                # Reset to 0 to ensure no residual LoRA effects
                module.lora_strength = 0.0
