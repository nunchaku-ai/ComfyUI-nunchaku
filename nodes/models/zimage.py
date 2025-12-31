"""
This module provides the :class:`NunchakuZImageDiTLoader` class for loading Nunchaku Z-Image models.
"""

import json
import logging

import comfy.utils
import torch
from comfy import model_detection, model_management

from nunchaku.models.transformers.utils import convert_fp16, patch_scale_key
from nunchaku.utils import check_hardware_compatibility, get_precision_from_quantization_config, is_turing

from ...model_configs.zimage import NunchakuZImage
from ...model_patcher.zimage import ZImageModelPatcher
from ..utils import get_filename_list, get_full_path_or_raise


def _patch_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Patch the state dict.

    Convert the keys from diffusers style to Comfy-Org style keys.
    See https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors

    Parameters
    ----------
    state_dict (dict[str, torch.Tensor]):
        the state dict loaded from diffuser style safetensors checkpoint file.

    Returns
    -------
    dict[str, torch.Tensor]:
        the patched state dict which follows Comfy-Org style keys.
    """
    patched_state_dict = {}
    quant_sub_keys = ["wscales", "wcscales", "wtscale", "smooth_factor_orig", "smooth_factor", "proj_down", "proj_up"]

    for key, value in state_dict.items():
        # region: attention
        ## case for quantized attention qkv
        if "attention.to_qkv" in key:
            patched_state_dict[key.replace("to_qkv", "qkv")] = value
        ## case for non-quantized attetion qkv
        elif "attention.to_q" in key:
            q_weight = state_dict[key]
            k_weight = state_dict[key.replace("to_q", "to_k")]
            v_weight = state_dict[key.replace("to_q", "to_v")]
            patched_state_dict[key.replace("to_q", "qkv")] = torch.cat([q_weight, k_weight, v_weight], dim=0)
        elif "attention.to_k" in key or "attention.to_v" in key:
            continue
        ## case for attention out
        elif "attention.to_out" in key:
            patched_state_dict[key.replace("to_out.0", "out")] = value
        # end of region
        # region: feed forward
        ## case for quantized feed forward
        elif "feed_forward.net.0.proj.qweight" in key:
            patched_state_dict[key.replace("net.0.proj", "w13")] = value
            for subkey in quant_sub_keys:
                quant_param_key = key.replace("qweight", subkey)
                if quant_param_key in state_dict:
                    patched_state_dict[quant_param_key.replace("net.0.proj", "w13")] = state_dict[quant_param_key]
        elif any("feed_forward.net.0.proj." + subkey in key for subkey in quant_sub_keys):
            continue
        elif "feed_forward.net.2.qweight" in key:
            patched_state_dict[key.replace("net.2", "w2")] = value
            for subkey in quant_sub_keys:
                quant_param_key = key.replace("qweight", subkey)
                if quant_param_key in state_dict:
                    patched_state_dict[quant_param_key.replace("net.2", "w2")] = state_dict[quant_param_key]
        elif any("feed_forward.net.2." + subkey in key for subkey in quant_sub_keys):
            continue
        ## case for non-quantized feed forward
        elif "feed_forward.net.0.proj.weight" in key:
            w3, w1 = torch.chunk(value, chunks=2, dim=0)
            w2 = state_dict[key.replace("0.proj", "2")]
            patched_state_dict[key.replace("net.0.proj", "w1")] = w1
            patched_state_dict[key.replace("net.0.proj", "w2")] = w2
            patched_state_dict[key.replace("net.0.proj", "w3")] = w3
        elif "feed_forward.net.2.weight" in key:
            continue
        # end of region
        # region: others
        elif "attention.norm_q" in key:
            patched_state_dict[key.replace("norm_q", "q_norm")] = value
        elif "attention.norm_k" in key:
            patched_state_dict[key.replace("norm_k", "k_norm")] = value
        elif "all_final_layer.2-1" in key:
            patched_state_dict[key.replace("all_final_layer.2-1", "final_layer")] = value
        elif "all_x_embedder.2-1" in key:
            patched_state_dict[key.replace("all_x_embedder.2-1", "x_embedder")] = value
        else:
            patched_state_dict[key] = value
        # end of region

    return patched_state_dict


def _load(sd: dict[str, torch.Tensor], metadata: dict[str, str] = {}):
    """
    Load a Nunchaku-quantized Z-Image diffusion model.

    Parameters
    ----------
    sd : dict[str, torch.Tensor]
        The state dictionary of the model.
    metadata : dict[str, str], optional
        Metadata containing quantization configuration (default is empty dict).

    Returns
    -------
    comfy.model_patcher.ModelPatcher
        The patched and loaded Qwen-Image model ready for inference.
    """
    quantization_config = json.loads(metadata.get("quantization_config", "{}"))
    precision = get_precision_from_quantization_config(quantization_config)
    rank = quantization_config.get("rank", 32)
    skip_refiners = quantization_config.get("skip_refiners", False)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)

    load_device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()
    check_hardware_compatibility(quantization_config, load_device)

    new_sd = sd

    model_config = NunchakuZImage(rank=rank, precision=precision, skip_refiners=skip_refiners)

    if not is_turing():
        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        unet_dtype = model_management.unet_dtype(
            model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype
        )
        manual_cast_dtype = model_management.unet_manual_cast(
            unet_dtype, load_device, model_config.supported_inference_dtypes
        )
        torch_dtype = torch.bfloat16
    else:
        unet_dtype = torch.bfloat16
        manual_cast_dtype = torch.float16
        torch_dtype = torch.float16
    logging.info(f"unet_dtype: {unet_dtype}, manual_cast_dtype: {manual_cast_dtype}, svdq_linear_dtype: {torch_dtype}")
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    patched_sd = _patch_state_dict(new_sd)
    model = model_config.get_model(patched_sd, "", torch_dtype=torch_dtype)

    patch_scale_key(model.diffusion_model, patched_sd)
    if torch_dtype == torch.float16:
        convert_fp16(model.diffusion_model, patched_sd)

    model.load_model_weights(patched_sd, "")
    return ZImageModelPatcher(model, load_device=load_device, offload_device=offload_device)


class NunchakuZImageDiTLoader:
    """
    Loader for Nunchaku Z-Image models.

    Attributes
    ----------
    RETURN_TYPES : tuple
        Output types for the node ("MODEL",).
    FUNCTION : str
        Name of the function to call ("load_model").
    CATEGORY : str
        Node category ("Nunchaku").
    TITLE : str
        Node title ("Nunchaku Z-Image DiT Loader").
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        return {
            "required": {
                "model_name": (
                    get_filename_list("diffusion_models"),
                    {"tooltip": "The Nunchaku Z-Image model."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Z-Image DiT Loader"

    def load_model(self, model_name: str, **kwargs):
        """
        Load the Z-Image model from file and return a patched model.

        Parameters
        ----------
        model_name : str
            The filename of the Z-Image model to load.

        Returns
        -------
        tuple
            A tuple containing the loaded and patched model.
        """
        model_path = get_full_path_or_raise("diffusion_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)

        model = _load(sd, metadata=metadata)

        return (model,)
