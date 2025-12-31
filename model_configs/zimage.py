"""
This module provides a wrapper for ComfyUI's Z-Image model configuration.

Note
----
Codes are adapted from:
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_detection.py#model_config_from_unet
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py#ZImage
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py#Lumina2
"""

import torch
from comfy.model_base import Lumina2
from comfy.supported_models import ZImage as ZImageModelConfig

from ..models.zimage import patch_model


class NunchakuZImage(ZImageModelConfig):
    """
    Nunchaku Z-Image model_config.
    """

    _DIT_CONFIG_ = {
        # This dict is the same as the result of execution of `comfy.model_detection#detect_unet_config` method for Z-Image
        "image_model": "lumina2",
        "patch_size": 2,
        "in_channels": 16,
        "cap_feat_dim": 2560,
        "n_layers": 30,
        "qk_norm": True,
        "dim": 3840,
        "n_heads": 30,
        "n_kv_heads": 30,
        "axes_dims": [32, 48, 48],
        "axes_lens": [1536, 512, 512],
        "rope_theta": 256.0,
        "ffn_dim_multiplier": (8.0 / 3.0),
        "z_image_modulation": True,
        "time_scale": 1000.0,
        "pad_tokens_multiple": 32,
    }

    def __init__(self, rank: int = 32, precision: str = "int4", skip_refiners: bool = False):
        super().__init__(unet_config=self._DIT_CONFIG_)
        self.rank = rank
        self.precision = precision
        self.skip_refiners = skip_refiners

    def get_model(self, state_dict: dict[str, torch.Tensor], prefix: str = "", device=None, **kwargs) -> Lumina2:
        """
        Instantiate and return a Nunchaku optimized Lumina2 model_base object.

        Parameters
        ----------
        state_dict : dict
            Model state dictionary.
        prefix : str, optional
            Prefix for parameter names (default is "").
        device : torch.device or str, optional
            Device to load the model onto.
        **kwargs
            Additional keyword arguments for model initialization.

        Returns
        -------
        model_base.Lumina2
            Instantiated model_base.Lumina2 object with Nunchaku quantized transformer blocks.
        """
        out: Lumina2 = super().get_model(state_dict, prefix, device)
        patch_model(
            out.diffusion_model, skip_refiners=self.skip_refiners, rank=self.rank, precision=self.precision, **kwargs
        )
        return out
