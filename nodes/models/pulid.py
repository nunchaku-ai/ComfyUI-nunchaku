"""
This module provides nodes and utilities for integrating the Nunchaku PuLID pipeline
with ComfyUI, enabling face restoration and enhancement using PuLID and related models.

.. note::

    Adapted from: https://github.com/lldacing/ComfyUI_PuLID_Flux_ll
"""

import logging
import os
from functools import partial
from types import MethodType

import comfy
import numpy as np
import torch

from nunchaku.models.pulid.pulid_forward import pulid_forward
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDPipeline

from ...wrappers.flux import ComfyFluxWrapper, copy_with_ctx
from ..utils import folder_paths, get_filename_list, get_full_path_or_raise
from .utils import set_extra_config_model_path

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


set_extra_config_model_path("pulid", "pulid")
set_extra_config_model_path("insightface", "insightface")
set_extra_config_model_path("facexlib", "facexlib")


class NunchakuFluxPuLIDApplyV2:
    """
    Node for applying PuLID to a Nunchaku FLUX model.
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
                "model": ("MODEL",),
                "pulid_pipline": ("PULID_PIPELINE",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "attn_mask": ("MASK",),
                "options": ("OPTIONS",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku FLUX PuLID Apply V2"

    def apply(
        self,
        model,
        pulid_pipline: PuLIDPipeline,
        image,
        weight: float,
        start_at: float,
        end_at: float,
        attn_mask=None,
        options=None,
        unique_id=None,
    ):
        """
        Apply PuLID ID customization according to the given image to the model.

        Parameters
        ----------
        model : object
            The Nunchaku FLUX model to modify.
        pulid_pipline : :class:`~nunchaku.pipeline.pipeline_flux_pulid.PuLIDPipeline`
            The PuLID pipeline instance.
        image : np.ndarray or torch.Tensor
            The input image for identity embedding extraction.
        weight : float
            The strength of the identity guidance.
        start_at : float
            The starting timestep for applying the effect.
        end_at : float
            The ending timestep for applying the effect.
        attn_mask : optional
            Not supported for now.
        options : optional
            Additional options (unused).
        unique_id : optional
            Unique identifier (unused).

        Returns
        -------
        tuple
            A tuple containing the modified model.

        Raises
        ------
        NotImplementedError
            If attn_mask is provided.
        """
        all_embeddings = []
        for i in range(image.shape[0]):
            single_image = image[i : i + 1].squeeze().cpu().numpy() * 255.0
            single_image = np.clip(single_image, 0, 255).astype(np.uint8)

            id_embedding, _ = pulid_pipline.get_id_embedding(single_image)
            if id_embedding is not None:
                all_embeddings.append(id_embedding)

        if not all_embeddings:
            logger.warning("Nunchaku PuLID: No face detected in any of the images. Skipping PuLID.")
            return (model,)

        id_embeddings = torch.mean(torch.stack(all_embeddings), dim=0)

        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfyFluxWrapper)

        ret_model_wrapper, ret_model = copy_with_ctx(model_wrapper)

        ret_model_wrapper.pulid_pipeline = pulid_pipline
        ret_model_wrapper.customized_forward = partial(
            pulid_forward, id_embeddings=id_embeddings, id_weight=weight, start_timestep=start_at, end_timestep=end_at
        )

        if attn_mask is not None:
            raise NotImplementedError("Attn mask is not supported for now in Nunchaku FLUX PuLID Apply V2.")

        return (ret_model,)


class NunchakuPuLIDLoaderV2:
    """
    Node for loading the PuLID pipeline.

    This node loads the PuLID model, EVA CLIP model, and required face libraries, and
    returns both the original model and a ready-to-use PuLID pipeline.
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
        pulid_files = get_filename_list("pulid")
        clip_files = get_filename_list("clip")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The nunchaku model."}),
                "pulid_file": (pulid_files, {"tooltip": "Path to the PuLID model."}),
                "eva_clip_file": (clip_files, {"tooltip": "Path to the EVA clip model."}),
                "insight_face_provider": (["gpu", "cpu"], {"default": "gpu", "tooltip": "InsightFace ONNX provider."}),
            }
        }

    RETURN_TYPES = ("MODEL", "PULID_PIPELINE")
    FUNCTION = "load"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku PuLID Loader V2"

    def load(self, model, pulid_file: str, eva_clip_file: str, insight_face_provider: str):
        """
        Load the PuLID pipeline and associate it with the given Nunchaku FLUX model.

        Parameters
        ----------
        model : object
            The Nunchaku FLUX model to use.
        pulid_file : str
            Path to the PuLID model file.
        eva_clip_file : str
            Path to the EVA CLIP model file.
        insight_face_provider : str
            ONNX provider for InsightFace ("gpu" or "cpu").

        Returns
        -------
        tuple
            (model, pulid_pipeline)
        """
        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfyFluxWrapper)
        transformer = model_wrapper.model

        device = comfy.model_management.get_torch_device()
        weight_dtype = next(transformer.parameters()).dtype

        pulid_path = get_full_path_or_raise("pulid", pulid_file)
        eva_clip_path = get_full_path_or_raise("clip", eva_clip_file)
        insightface_dirpath = folder_paths.get_folder_paths("insightface")[0]
        facexlib_dirpath = folder_paths.get_folder_paths("facexlib")[0]

        pulid_pipline = PuLIDPipeline(
            dit=transformer,
            device=device,
            weight_dtype=weight_dtype,
            onnx_provider=insight_face_provider,
            pulid_path=pulid_path,
            eva_clip_path=eva_clip_path,
            insightface_dirpath=insightface_dirpath,
            facexlib_dirpath=facexlib_dirpath,
        )

        return (model, pulid_pipline)


class NunchakuPulidApply:
    """
    Deprecated node for applying PuLID to a Nunchaku FLUX model.

    Attributes
    ----------
    pulid_device : str
        The device to use for PuLID inference (default: "cuda").
    weight_dtype : torch.dtype
        The data type for model weights (default: torch.bfloat16).
    onnx_provider : str
        The ONNX provider for InsightFace ("gpu" or "cpu", default: "gpu").
    pretrained_model : object or None
        The loaded PuLID model, if any.

    .. warning::
        This node is deprecated and will be removed in December 2025.
        Please use :class:`NunchakuFluxPuLIDApplyV2` instead.
    """

    def __init__(self):
        self.pulid_device = "cuda"
        self.weight_dtype = torch.bfloat16
        self.onnx_provider = "gpu"
        self.pretrained_model = None

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
                "pulid": ("PULID", {"tooltip": "from Nunchaku Pulid Loader"}),
                "image": ("IMAGE", {"tooltip": "The image to encode"}),
                "model": ("MODEL", {"tooltip": "The nunchaku model."}),
                "ip_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "ip_weight",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Pulid Apply (Deprecated)"

    def apply(self, pulid, image, model, ip_weight):
        """
        Apply PuLID identity embeddings to the given Nunchaku FLUX model.

        Parameters
        ----------
        pulid : object
            The PuLID pipeline instance.
        image : torch.Tensor
            The image to encode for identity.
        model : object
            The Nunchaku FLUX model.
        ip_weight : float
            The weight for the identity embedding.

        Returns
        -------
        tuple
            The updated model with PuLID applied.
        """
        logger.warning(
            'This node is deprecated and will be removed in December 2025. Directly use "Nunchaku FLUX PuLID Apply V2" instead.'
        )

        image = image.squeeze().cpu().numpy() * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        id_embeddings, _ = pulid.get_id_embedding(image)
        model.model.diffusion_model.model.forward = MethodType(
            partial(pulid_forward, id_embeddings=id_embeddings, id_weight=ip_weight), model.model.diffusion_model.model
        )
        return (model,)


class NunchakuPulidLoader:
    """
    Deprecated node for loading the PuLID pipeline for a Nunchaku FLUX model.

    .. warning::
        This node is deprecated and will be removed in December 2025.
        Use :class:`NunchakuPuLIDLoaderV2` instead.

    Attributes
    ----------
    pulid_device : str
        Device to load the PuLID pipeline on (default: "cuda").
    weight_dtype : torch.dtype
        Data type for model weights (default: torch.bfloat16).
    onnx_provider : str
        ONNX provider to use (default: "gpu").
    pretrained_model : str or None
        Path to the pretrained PuLID model, if any.
    """

    def __init__(self):
        """
        Initialize the loader with default device, dtype, and ONNX provider.
        """
        self.pulid_device = "cuda"
        self.weight_dtype = torch.bfloat16
        self.onnx_provider = "gpu"
        self.pretrained_model = None

    @classmethod
    def INPUT_TYPES(s):
        """
        Returns the required input types for this node.

        Returns
        -------
        dict
            Dictionary specifying required inputs.
        """
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The nunchaku model."}),
            }
        }

    RETURN_TYPES = ("MODEL", "PULID")
    FUNCTION = "load"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Pulid Loader (Deprecated)"

    def load(self, model):
        """
        Load the PuLID pipeline for the given Nunchaku FLUX model.

        .. warning::
            This node is deprecated and will be removed in December 2025.
            Use :class:`NunchakuPuLIDLoaderV2` instead.

        Parameters
        ----------
        model : object
            The Nunchaku FLUX model.

        Returns
        -------
        tuple
            The input model and the loaded PuLID pipeline.
        """
        logger.warning(
            'This node is deprecated and will be removed in December 2025. Directly use "Nunchaku PuLID Loader V22 instead.'
        )
        pulid_model = PuLIDPipeline(
            dit=model.model.diffusion_model.model,
            device=self.pulid_device,
            weight_dtype=self.weight_dtype,
            onnx_provider=self.onnx_provider,
        )
        pulid_model.load_pretrain(self.pretrained_model)

        return (model, pulid_model)
