"""
This module provides the :class:`NunchakuZImageLoraLoader` node
for applying LoRA weights to Nunchaku Z-Image models within ComfyUI.
"""

import logging
import os

from ...wrappers.zimage import ComfyZImageWrapper, copy_with_ctx, reset_lora
from ..utils import get_filename_list, get_full_path_or_raise

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuZImageLoraLoader:
    """
    Node for loading and applying a LoRA to a Nunchaku Z-Image model.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRA will be applied to. "
                        "Make sure the model is loaded by `Nunchaku Z-Image DiT Loader`."
                    },
                ),
                "lora_name": (
                    get_filename_list("loras"),
                    {"tooltip": "The file name of the LoRA."},
                ),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Z-Image LoRA Loader"

    def load_lora(self, model, lora_name: str, lora_strength: float):
        if abs(lora_strength) < 1e-5:
            model_wrapper = model.model.diffusion_model
            base_model = model_wrapper.model if isinstance(model_wrapper, ComfyZImageWrapper) else model_wrapper
            reset_lora(base_model)
            if isinstance(model_wrapper, ComfyZImageWrapper):
                model_wrapper.loras = []
                model_wrapper._applied_loras = []
            return (model,)

        model_wrapper = model.model.diffusion_model

        # If it's not wrapped in our custom wrapper yet, wrap it
        if not isinstance(model_wrapper, ComfyZImageWrapper):
            # We need the model_config and other ctx from the model
            # This requires the loader to have populated ctx_for_copy
            if not hasattr(model_wrapper, "ctx_for_copy") or not model_wrapper.ctx_for_copy:
                # If we can't find the context, we might be dealing with a raw model
                # Try to infer or check if it's already a Nunchaku model
                logger.warning("Model wrapper does not have ctx_for_copy. Attempting to wrap anyway.")

            # Create the wrapper
            # Note: We expect the input 'model' (ModelPatcher) to have the necessary info
            # In our case, we'll update the loader next to ensure it's there.
            config = getattr(model_wrapper, "config", {})
            ctx = getattr(model_wrapper, "ctx_for_copy", {})

            model_wrapper = ComfyZImageWrapper(model_wrapper, config=config, ctx_for_copy=ctx)
            model.model.diffusion_model = model_wrapper

        lora_path = get_full_path_or_raise("loras", lora_name)

        ret_model_wrapper, ret_model = copy_with_ctx(model_wrapper)
        ret_model_wrapper.loras = [*model_wrapper.loras, (lora_path, lora_strength)]

        return (ret_model,)
