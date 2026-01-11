"""
This module provides the :class:`NunchakuQwenImageLoraLoader` node
for applying LoRA weights to Nunchaku Qwen Image models within ComfyUI.
"""

import copy
import logging
import os

import folder_paths

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuQwenImageLoraLoader:
    """
    Node for loading and applying a LoRA to a Nunchaku Qwen Image model.
    """
    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRA will be applied to. "
                        "Make sure the model is loaded by `Nunchaku Qwen Image DiT Loader`."
                    },
                ),
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
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
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "Nunchaku Qwen Image LoRA Loader"
    CATEGORY = "Nunchaku"
    DESCRIPTION = "LoRAs are used to modify the diffusion model, altering the way in which latents are denoised."

    def load_lora(self, model, lora_name: str, lora_strength: float):
        if abs(lora_strength) < 1e-5:
            return (model,)

        model_wrapper = model.model.diffusion_model

        from ...wrappers.qwenimage import ComfyQwenImageWrapper
        if not isinstance(model_wrapper, ComfyQwenImageWrapper):
            logger.error("❌ Model type mismatch! Please use 'Nunchaku Qwen Image DiT Loader'.")
            raise TypeError(f"This LoRA loader only works with Nunchaku Qwen Image models, but got {type(model_wrapper).__name__}.")

        transformer = model_wrapper.model

        # Flux-style deepcopy
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)
        ret_model_wrapper = ret_model.model.diffusion_model
        model_wrapper.model = transformer
        ret_model_wrapper.model = transformer

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        ret_model_wrapper.loras.append((lora_path, lora_strength))

        logger.info(f"LoRA added: {lora_name} (strength={lora_strength})")
        logger.debug(f"Total LoRAs: {len(ret_model_wrapper.loras)}")

        return (ret_model,)


class NunchakuQwenImageLoraStack:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Qwen Image model.
    """
    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model to apply LoRAs to."},
                ),
            },
            "optional": {},
        }
        for i in range(1, 6):
            inputs["optional"][f"lora_name_{i}"] = (["None"] + folder_paths.get_filename_list("loras"),)
            inputs["optional"][f"lora_strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.",)
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku Qwen Image LoRA Stack"
    CATEGORY = "Nunchaku"
    DESCRIPTION = "Apply multiple LoRAs to a diffusion model in a single node."

    def load_lora_stack(self, model, **kwargs):
        loras_to_apply = []
        for i in range(1, 6):
            lora_name = kwargs.get(f"lora_name_{i}")
            lora_strength = kwargs.get(f"lora_strength_{i}", 1.0)
            if lora_name and lora_name != "None" and abs(lora_strength) > 1e-5:
                loras_to_apply.append((lora_name, lora_strength))

        if not loras_to_apply:
            return (model,)

        model_wrapper = model.model.diffusion_model

        from ...wrappers.qwenimage import ComfyQwenImageWrapper
        if not isinstance(model_wrapper, ComfyQwenImageWrapper):
            logger.error("❌ Model type mismatch! Please use 'Nunchaku Qwen Image DiT Loader'.")
            raise TypeError(f"This LoRA loader only works with Nunchaku Qwen Image models, but got {type(model_wrapper).__name__}.")

        transformer = model_wrapper.model

        # Flux-style deepcopy
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)
        ret_model_wrapper = ret_model.model.diffusion_model
        model_wrapper.model = transformer
        ret_model_wrapper.model = transformer

        ret_model_wrapper.loras = model_wrapper.loras.copy()

        for lora_name, lora_strength in loras_to_apply:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            ret_model_wrapper.loras.append((lora_path, lora_strength))
            logger.debug(f"LoRA added to stack: {lora_name} (strength={lora_strength})")

        logger.info(f"Total LoRAs in stack: {len(ret_model_wrapper.loras)}")

        return (ret_model,)