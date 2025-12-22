"""
This module provides the :class:`NunchakuZImageLoraLoader` node
for applying LoRA weights to Nunchaku Z-Image models within ComfyUI.
"""

import logging

import torch
import torch.nn as nn

# from comfy.lora import load_lora, model_lora_keys_unet
from comfy.model_base import BaseModel
from comfy.utils import get_attr  # load_torch_file
from comfy.weight_adapter.lora import LoRAAdapter

from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight, unpack_lowrank_weight
from nunchaku.models.linear import SVDQW4A4Linear

from ..utils import get_filename_list  # get_full_path_or_raise


def concat_lora_weights(
    base_down: torch.Tensor,
    base_up: torch.Tensor,
    new_downs: list[torch.Tensor],
    new_ups: list[torch.Tensor],
    strengths: list[float],
):
    """
    Concatenate multiple LoRA weights into single down and up weights.

    Parameters
    ----------
    base_down : torch.Tensor
        The base LoRA down weight.
    base_up : torch.Tensor
        The base LoRA up weight.
    new_downs : list of torch.Tensor
        List of new LoRA down weights to concatenate.
    new_ups : list of torch.Tensor
        List of new LoRA up weights to concatenate.
    strengths : list of float
        List of strength/scale factors for each new LoRA.

    Returns
    -------
    tuple of torch.Tensor
        The concatenated down and up weights.
    """
    assert len(new_downs) == len(new_ups) == len(strengths), "Lengths of new_downs, new_ups, and strengths must match."
    assert (base_down is None) == (base_up is None), "Both base_down and base_up should be None or not None."

    combined_new_downs = torch.cat([nd * s for nd, s in zip(new_downs, strengths)], dim=0)
    combined_new_ups = torch.block_diag(*new_ups)

    if base_down is None:
        return combined_new_downs, combined_new_ups

    assert base_up.shape[1] == base_down.shape[0], "Base up and down weights shapes do not match."
    assert base_up.shape[0] == sum(
        new_up.shape[0] for new_up in new_ups
    ), "Total new up weights rows do not match base up weight rows."
    assert all(
        new_down.shape[1] == base_down.shape[1] for new_down in new_downs
    ), "New up and down weights shapes do not match."

    concatenated_down = torch.cat([base_down, combined_new_downs], dim=0)
    concatenated_up = torch.cat([base_up, combined_new_ups], dim=1)
    return concatenated_down, concatenated_up


UNSUPPORTED_PARTS = ["q_norm", "k_norm", "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"]


NON_QUANTIZED_PARTS = ["adaLN_modulation"]


def compose_z_image_loras(
    loras: list[tuple[str, dict[str | tuple, LoRAAdapter], float]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Compose multiple LoRA adapters into a single LoRA adapter for Nunchaku Z-Image models.

    Parameters
    ----------
    loras : list of (str, LoRAAdapter, float)
        Each tuple contains:
            - Path to a LoRA safetensors file.
            - Loaded dict of LoRAAdapters.
            - Strength/scale factor for that LoRA.

    Returns
    -------
    dict[str, torch.Tensor]
        The composed LoRA state dict.
    """
    composed_dict = {}
    for lora_full_path, lora_adapters, strength in loras:
        # `lora_full_path` is unused currently. It will be used as a global key for avoiding duplicated loading in the future.
        logging.info(f">>> composing LoRA from {lora_full_path} with strength {strength}")
        for adapter_key, lora_adapter in lora_adapters.items():
            if isinstance(adapter_key, str):
                sd_key = adapter_key
            elif isinstance(adapter_key, tuple):
                sd_key = adapter_key[0]
                assert "qkv" in sd_key, f"Unexpected adapter_key tuple: {adapter_key}"
                offset = adapter_key[1]
            else:
                raise ValueError(f"Invalid adapter_key type: {type(adapter_key)}")

            assert sd_key.startswith("diffusion_model.layers.") and sd_key.endswith(
                ".weight"
            ), f"Unexpected LoRA key: {sd_key}"
            assert not any(
                part in sd_key for part in UNSUPPORTED_PARTS
            ), f"LoRA key {sd_key} contains unsupported parts."

            composed_down, composed_up = composed_dict.get(sd_key, (None, None))
            lora_up, lora_down, alpha, mid, dora_scale, reshape = lora_adapter.weights
            assert (
                lora_up.shape[1] == lora_down.shape[0]
            ), f"LoRA up and down shapes do not match for key {sd_key}: {lora_up.shape}, {lora_down.shape}"
            rank = lora_down.shape[0]
            if ".qkv." in sd_key:
                if offset[1] != 0:
                    continue
                # logging.info(f">>> Composing fused qkv LoRA key: {sd_key}")
                q_up, q_down, q_alpha, q_rank = lora_up, lora_down, alpha, rank
                k_adapter = lora_adapters[(sd_key, (offset[0], offset[1] + offset[2], offset[2]))]
                k_up, k_down, k_alpha, k_mid, k_dora_scale, k_reshape = k_adapter.weights
                k_rank = k_down.shape[0]
                v_adapter = lora_adapters[(sd_key, (offset[0], offset[1] + offset[2] * 2, offset[2]))]
                v_up, v_down, v_alpha, v_mid, v_dora_scale, v_reshape = v_adapter.weights
                v_rank = v_down.shape[0]

                composed_down, composed_up = concat_lora_weights(
                    composed_down,
                    composed_up,
                    [q_down, k_down, v_down],
                    [q_up, k_up, v_up],
                    [
                        strength * (q_alpha / q_rank) if q_alpha is not None else strength,
                        strength * (k_alpha / k_rank) if k_alpha is not None else strength,
                        strength * (v_alpha / v_rank) if v_alpha is not None else strength,
                    ],
                )
                composed_dict[sd_key] = (composed_down, composed_up)
            elif ".feed_forward.w1." in sd_key:
                # logging.info(f">>> Composing fused ff LoRA key: {sd_key}")
                w1_up, w1_down, w1_alpha, w1_rank = lora_up, lora_down, alpha, rank
                w3_adapter = lora_adapters[(sd_key.replace("w1", "w3"))]
                w3_up, w3_down, w3_alpha, w3_mid, w3_dora_scale, w3_reshape = w3_adapter.weights
                w3_rank = w3_down.shape[0]

                composed_down, composed_up = concat_lora_weights(
                    composed_down,
                    composed_up,
                    [w3_down, w1_down],
                    [w3_up, w1_up],
                    [
                        strength * (w3_alpha / w3_rank) if w3_alpha is not None else strength,
                        strength * (w1_alpha / w1_rank) if w1_alpha is not None else strength,
                    ],
                )
                composed_dict[sd_key.replace("w1", "w13")] = (composed_down, composed_up)
            elif ".feed_forward.w3." in sd_key:
                continue
            elif (
                ".out." in sd_key
                or ".feed_forward.w2." in sd_key
                or any(part in sd_key for part in NON_QUANTIZED_PARTS)
            ):
                # elif ".out." in sd_key or ".feed_forward.w" in sd_key or any(part in sd_key for part in NON_QUANTIZED_PARTS):
                # logging.info(f">>> Composing non fused LoRA key: {sd_key}")
                composed_down, composed_up = concat_lora_weights(
                    composed_down,
                    composed_up,
                    [lora_down],
                    [lora_up],
                    [strength * (alpha / rank) if alpha is not None else strength],
                )
                composed_dict[sd_key] = (composed_down, composed_up)
            else:
                raise ValueError(f"Unexpected LoRA key: {sd_key}")
    # TODO verify correctness
    logging.info(f">>> Finished composing LoRAs for Nunchaku Z-Image. Size: {len(composed_dict)}")
    return composed_dict


class NunchakuZImageComposedLoRAs:
    def __init__(self, composed_lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}):
        self.composed_lora_weights = composed_lora_weights


def apply_lora_to_svdq_linear(linear: SVDQW4A4Linear, lora_down: torch.Tensor, lora_up: torch.Tensor):
    dtype = linear.proj_down.dtype
    proj_down = unpack_lowrank_weight(linear.proj_down, down=True)
    proj_up = unpack_lowrank_weight(linear.proj_up, down=False)
    # _factor = 1.0 / linear.smooth_factor.to(dtype=lora_down.dtype)
    # concatenated_down = torch.cat([proj_down, _factor[None, :] * lora_down], dim=0).to(dtype=dtype)
    concatenated_down = torch.cat([proj_down, lora_down], dim=0).to(dtype=dtype)
    concatenated_up = torch.cat([proj_up, lora_up], dim=1).to(dtype=dtype)

    linear.proj_down = nn.Parameter(pack_lowrank_weight(concatenated_down, down=True))
    linear.proj_up = nn.Parameter(pack_lowrank_weight(concatenated_up, down=False))


def apply_lora_to_zimage_transformer(
    model: BaseModel, composed_lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]]
):
    """
    Apply a LoRA adapter to a Nunchaku Z-Image NextDiT transformer model.

    Parameters
    ----------
    model : BaseModel
        The base model containing the NextDiT transformer which contains `SVDQLinear` modules.
    composed_lora_weights : dict[str, tuple[torch.Tensor, torch.Tensor]]
        The composed LoRA weights to apply.
    """
    for weight_key, (lora_down, lora_up) in composed_lora_weights.items():
        attr_name = weight_key.replace(".weight", "")
        linear = get_attr(model, attr_name)
        if isinstance(linear, nn.Linear):
            logging.info(
                f">>> Applying LoRA to non-quantized linear at {attr_name}, linear weight shape: {linear.weight.shape}, lora_down shape: {lora_down.shape}, lora_up shape: {lora_up.shape}"
            )
            linear.weight = nn.Parameter(linear.weight + torch.matmul(lora_up, lora_down).to(linear.weight.dtype))
            # TODO verify correctness
        elif isinstance(linear, SVDQW4A4Linear):
            logging.info(
                f">>> Applying LoRA to quantized linear at {attr_name}, proj_down shape: {linear.proj_down.shape}, proj_up shape: {linear.proj_up.shape}, lora_down shape: {lora_down.shape}, lora_up shape: {lora_up.shape}"
            )
            apply_lora_to_svdq_linear(linear, lora_down, lora_up)
            # TODO verify correctness
        else:
            raise ValueError(f"Unexpected linear type at {attr_name}: {type(linear)}")


class NunchakuZImageLoraLoader:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Z-Image model with dynamic input.

    This node allows you to configure multiple LoRAs with their respective strengths
    in a single node, no need to chain multiple LoRA loader nodes.

    Attributes
    ----------
    RETURN_TYPES : tuple
        The return type of the node ("MODEL",).
    OUTPUT_TOOLTIPS : tuple
        Tooltip for the output.
    FUNCTION : str
        The function to call ("load_multiple_loras").
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
        Defines the input types for the this node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and optional LoRA inputs.
        """
        # Base inputs
        # inputs = {
        #     "required": {
        #         "model": (
        #             "MODEL",
        #             {
        #                 "tooltip": "The Nunchaku Z-Image model the LoRAs will be applied to. "
        #                 "Make sure the model is loaded by `Nunchaku Z-Image DiT Loader`."
        #             },
        #         ),
        #     },
        #     "optional": {},
        # }
        inputs = {}
        for i in range(5):  # Support up to 5 LoRAs
            if i == 0:
                key = "required"
            else:
                key = "optional"
            inputs[key][f"lora_name_{i + 1}"] = (
                ["None"] + get_filename_list("loras"),
                {"tooltip": f"The file name of LoRA {i + 1}. Select 'None' to skip this slot."},
            )
            inputs[key][f"lora_strength_{i + 1}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"Strength for LoRA {i + 1}. This value can be negative.",
                },
            )
        return inputs

    RETURN_TYPES = ("LoRAs",)
    OUTPUT_TOOLTIPS = ("Composed LoRA for Nunchaku Z-Image model.",)
    FUNCTION = "load_multiple_loras"
    TITLE = "Nunchaku Z-Image LoRA Loader"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "Load multiple LoRAs and compose them for Nunchaku Z-Image model. "
        "Supports at most 5 LoRAs simultaneously. Set unused slots to 'None' to skip them."
    )

    def load_multiple_loras(self, **kwargs):
        """
        Load and compose multiple LoRAs for a Nunchaku Z-Image diffusion model.

        Parameters
        ----------
        **kwargs
            Multiple LoRA names and strengths.

        Returns
        -------
        tuple
            A tuple containing the composed LoRA for Nunchaku Z-Image model.
        """
        # Collect LoRA information to apply
        loras_to_apply = []
        for i in range(1, 6):  # Check all 5 LoRA slots
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
            logging.info(">>> No LoRAs to apply, returning empty composed LoRA.")
            return (NunchakuZImageComposedLoRAs(),)

        # key_map is a dict, mapping diffuser style state dict keys to comfy style state dict keys.
        # key_map = model_lora_keys_unet(model.model)

        loras = []
        # for lora_name, lora_strength in loras_to_apply:
        #     lora_full_path = get_full_path_or_raise("loras", lora_name)
        #     lora_state_dict = load_torch_file(lora_full_path, safe_load=True)
        #     lora_adapters: dict[str | tuple, LoRAAdapter] = load_lora(lora_state_dict, key_map)
        #     logging.info(f">>> Loaded LoRA from {lora_full_path} with {len(lora_adapters)} adapters.")
        #     loras.append((lora_full_path, lora_adapters, lora_strength))

        composed_lora_weights = compose_z_image_loras(loras)

        # apply_lora_to_zimage_transformer(model.model, composed_lora_weights)

        return (NunchakuZImageComposedLoRAs(composed_lora_weights),)
