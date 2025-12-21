"""
This module provides the :class:`NunchakuFluxLoraLoader` node
for applying LoRA weights to Nunchaku FLUX models within ComfyUI.
"""

import logging
import os

from nunchaku.lora.flux import to_diffusers

from ...wrappers.flux import ComfyFluxWrapper, copy_with_ctx
from ..utils import get_filename_list, get_full_path_or_raise

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuFluxLoraLoader:
    """
    Node for loading and applying a LoRA to a Nunchaku FLUX model.

    Attributes
    ----------
    RETURN_TYPES : tuple
        The return type of the node ("MODEL",).
    OUTPUT_TOOLTIPS : tuple
        Tooltip for the output.
    FUNCTION : str
        The function to call ("load_lora").
    TITLE : str
        Node title.
    CATEGORY : str
        Node category.
    DESCRIPTION : str
        Node description.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRA will be applied to. "
                        "Make sure the model is loaded by `Nunchaku FLUX DiT Loader`."
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
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "Nunchaku FLUX LoRA Loader"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model, "
        "altering the way in which latents are denoised such as applying styles. "
        "You can link multiple LoRA nodes."
    )

    def load_lora(self, model, lora_name: str, lora_strength: float):
        """
        Apply a LoRA to a Nunchaku FLUX diffusion model.

        Parameters
        ----------
        model : object
            The diffusion model to modify.
        lora_name : str
            The name of the LoRA to apply.
        lora_strength : float
            The strength with which to apply the LoRA.

        Returns
        -------
        tuple
            A tuple containing the modified diffusion model.
        """
        if abs(lora_strength) < 1e-5:
            return (model,)  # If the strength is too small, return the original model

        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfyFluxWrapper)

        lora_path = get_full_path_or_raise("loras", lora_name)

        ret_model_wrapper, ret_model = copy_with_ctx(model_wrapper)

        ret_model_wrapper.loras = [*model_wrapper.loras, (lora_path, lora_strength)]
        sd = to_diffusers(lora_path)

        # To handle FLUX.1 tools LoRAs, which change the number of input channels
        if "transformer.x_embedder.lora_A.weight" in sd:
            new_in_channels = sd["transformer.x_embedder.lora_A.weight"].shape[1]
            assert new_in_channels % 4 == 0
            new_in_channels = new_in_channels // 4

            old_in_channels = ret_model.model.model_config.unet_config["in_channels"]
            if old_in_channels < new_in_channels:
                ret_model.model.model_config.unet_config["in_channels"] = new_in_channels

        return (ret_model,)


class NunchakuFluxLoraStack:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku FLUX model with dynamic input.

    This node allows you to configure multiple LoRAs with their respective strengths
    in a single node, providing the same effect as chaining multiple LoRA nodes.

    Attributes
    ----------
    RETURN_TYPES : tuple
        The return type of the node ("MODEL",).
    OUTPUT_TOOLTIPS : tuple
        Tooltip for the output.
    FUNCTION : str
        The function to call ("load_lora_stack").
    TITLE : str
        Node title.
    CATEGORY : str
        Node category.
    DESCRIPTION : str
        Node description.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the LoRA stack node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and optional LoRA inputs.
        """
        # Base inputs
        inputs = {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRAs will be applied to. "
                        "Make sure the model is loaded by `Nunchaku FLUX DiT Loader`."
                    },
                ),
            },
            "optional": {},
        }

        # Add fixed number of LoRA inputs (15 slots)
        for i in range(1, 16):  # Support up to 15 LoRAs
            inputs["optional"][f"lora_name_{i}"] = (
                ["None"] + get_filename_list("loras"),
                {"tooltip": f"The file name of LoRA {i}. Select 'None' to skip this slot."},
            )
            inputs["optional"][f"lora_strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"Strength for LoRA {i}. This value can be negative.",
                },
            )

        return inputs

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.",)
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku FLUX LoRA Stack"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "Apply multiple LoRAs to a diffusion model in a single node. "
        "Equivalent to chaining multiple LoRA nodes but more convenient for managing many LoRAs. "
        "Supports up to 15 LoRAs simultaneously. Set unused slots to 'None' to skip them."
    )

    def load_lora_stack(self, model, **kwargs):
        """
        Apply multiple LoRAs to a Nunchaku FLUX diffusion model.

        Parameters
        ----------
        model : object
            The diffusion model to modify.
        **kwargs
            Dynamic LoRA name and strength parameters.

        Returns
        -------
        tuple
            A tuple containing the modified diffusion model.
        """
        # Collect LoRA information to apply
        loras_to_apply = []

        for i in range(1, 16):  # Check all 15 LoRA slots
            lora_name = kwargs.get(f"lora_name_{i}")
            lora_strength = kwargs.get(f"lora_strength_{i}", 1.0)

            # Skip unset or None LoRAs
            if lora_name is None or lora_name == "None" or lora_name == "":
                continue

            # Skip LoRAs with zero strength
            if abs(lora_strength) < 1e-5:
                continue

            loras_to_apply.append((lora_name, lora_strength))

        # If no LoRAs need to be applied, return the original model
        if not loras_to_apply:
            return (model,)

        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfyFluxWrapper)

        ret_model_wrapper, ret_model = copy_with_ctx(model_wrapper)

        # Clear existing LoRA list
        ret_model_wrapper.loras = []

        # Track the maximum input channels needed
        max_in_channels = ret_model.model.model_config.unet_config["in_channels"]

        # Add all LoRAs
        for lora_name, lora_strength in loras_to_apply:
            lora_path = get_full_path_or_raise("loras", lora_name)
            ret_model_wrapper.loras.append((lora_path, lora_strength))

            # Check if input channels need to be updated
            sd = to_diffusers(lora_path)
            if "transformer.x_embedder.lora_A.weight" in sd:
                new_in_channels = sd["transformer.x_embedder.lora_A.weight"].shape[1]
                assert new_in_channels % 4 == 0
                new_in_channels = new_in_channels // 4
                max_in_channels = max(max_in_channels, new_in_channels)

        # Update the model's input channels
        if max_in_channels > ret_model.model.model_config.unet_config["in_channels"]:
            ret_model.model.model_config.unet_config["in_channels"] = max_in_channels

        return (ret_model,)
