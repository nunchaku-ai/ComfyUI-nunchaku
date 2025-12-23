"""
This module provides the :class:`NunchakuFluxDiTLoader` class for loading Nunchaku FLUX models.
It also supports attention implementation selection, CPU offload, and first-block caching.
"""

import gc
import json
import logging
import os

import comfy.model_management
import comfy.model_patcher
import torch
from comfy.supported_models import Flux, FluxSchnell

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_transformer
from nunchaku.utils import is_turing

from ...wrappers.flux import ComfyFluxWrapper
from ..utils import get_filename_list, get_full_path_or_raise

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuFluxDiTLoader:
    """
    Loader for Nunchaku FLUX.1 models.

    This class manages model loading, device selection, attention implementation,
    CPU offload, and caching for efficient inference.

    Attributes
    ----------
    transformer : :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel` or None
        The loaded transformer model.
    metadata : dict or None
        Metadata associated with the loaded model.
    model_path : str or None
        Path to the loaded model.
    device : torch.device or None
        Device on which the model is loaded.
    cpu_offload : bool or None
        Whether CPU offload is enabled.
    data_type : str or None
        Data type used for inference.
    patcher : object or None
        ComfyUI model patcher instance.
    """

    def __init__(self):
        """
        Initialize the NunchakuFluxDiTLoader.

        Sets up internal state and selects the default torch device.
        """
        self.transformer = None
        self.metadata = None
        self.model_path = None
        self.device = None
        self.cpu_offload = None
        self.data_type = None
        self.patcher = None
        self.device = comfy.model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        safetensor_files = get_filename_list("diffusion_models")

        ngpus = torch.cuda.device_count()

        all_turing = True
        for i in range(torch.cuda.device_count()):
            if not is_turing(f"cuda:{i}"):
                all_turing = False

        if all_turing:
            attention_options = ["nunchaku-fp16"]  # turing GPUs do not support flashattn2
            dtype_options = ["float16"]
        else:
            attention_options = ["nunchaku-fp16", "flash-attention2"]
            dtype_options = ["bfloat16", "float16"]

        return {
            "required": {
                "model_path": (
                    safetensor_files,
                    {"tooltip": "The Nunchaku FLUX model."},
                ),
                "cache_threshold": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1,
                        "step": 0.001,
                        "tooltip": "Adjusts the first-block caching tolerance"
                        "like `residual_diff_threshold` in WaveSpeed. "
                        "Increasing the value enhances speed at the cost of quality. "
                        "A typical setting is 0.12. Setting it to 0 disables the effect.",
                    },
                ),
                "attention": (
                    attention_options,
                    {
                        "default": attention_options[0],
                        "tooltip": (
                            "Attention implementation. The default implementation is `flash-attention2`. "
                            "`nunchaku-fp16` use FP16 attention, offering ~1.2× speedup. "
                            "Note that 20-series GPUs can only use `nunchaku-fp16`."
                        ),
                    },
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "Whether to enable CPU offload for the transformer model."
                        "auto' will enable it if the GPU memory is less than 14G.",
                    },
                ),
                "device_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": ngpus - 1,
                        "step": 1,
                        "display": "number",
                        "lazy": True,
                        "tooltip": "The GPU device ID to use for the model.",
                    },
                ),
                "data_type": (
                    dtype_options,
                    {
                        "default": dtype_options[0],
                        "tooltip": "Specifies the model's data type. Default is `bfloat16`. "
                        "For 20-series GPUs, which do not support `bfloat16`, use `float16` instead.",
                    },
                ),
            },
            "optional": {
                "i2f_mode": (
                    ["enabled", "always"],
                    {
                        "default": "enabled",
                        "tooltip": "The GEMM implementation for 20-series GPUs"
                        "— this option is only applicable to these GPUs.",
                    },
                )
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku FLUX DiT Loader"

    def load_model(
        self,
        model_path: str,
        attention: str,
        cache_threshold: float,
        cpu_offload: str,
        device_id: int,
        data_type: str,
        **kwargs,
    ):
        """
        Load a Nunchaku FLUX model with the specified configuration.

        Parameters
        ----------
        model_path : str
            Path to the model directory or safetensors file.
        attention : str
            Attention implementation to use ("nunchaku-fp16" or "flash-attention2").
        cache_threshold : float
            Caching tolerance for first-block cache. See :ref:`nunchaku:usage-fbcache` for details.
        cpu_offload : str
            Whether to enable CPU offload ("auto", "enable", "disable").
        device_id : int
            GPU device ID to use.
        data_type : str
            Data type for inference ("bfloat16" or "float16").
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tuple
            A tuple containing the loaded and patched model.
        """
        device = torch.device(f"cuda:{device_id}")

        model_path = get_full_path_or_raise("diffusion_models", model_path)

        # Check if the device_id is valid
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid device_id: {device_id}. Only {torch.cuda.device_count()} GPUs available.")

        # Get the GPU properties
        gpu_properties = torch.cuda.get_device_properties(device_id)
        gpu_memory = gpu_properties.total_memory / (1024**2)  # Convert to MiB
        gpu_name = gpu_properties.name
        logger.debug(f"GPU {device_id} ({gpu_name}) Memory: {gpu_memory} MiB")

        # Check if CPU offload needs to be enabled
        if cpu_offload == "auto":
            if gpu_memory < 14336:  # 14GB threshold
                cpu_offload_enabled = True
                logger.debug("VRAM < 14GiB, enabling CPU offload")
            else:
                cpu_offload_enabled = False
                logger.debug("VRAM > 14GiB, disabling CPU offload")
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
            logger.debug("Enabling CPU offload")
        else:
            cpu_offload_enabled = False
            logger.debug("Disabling CPU offload")

        if (
            self.model_path != model_path
            or self.device != device
            or self.cpu_offload != cpu_offload_enabled
            or self.data_type != data_type
        ):
            if self.transformer is not None:
                model_size = comfy.model_management.module_size(self.transformer)
                transformer = self.transformer
                self.transformer = None
                transformer.to("cpu")
                del transformer
                gc.collect()
                comfy.model_management.cleanup_models_gc()
                comfy.model_management.soft_empty_cache()
                comfy.model_management.free_memory(model_size, device)

            self.transformer, self.metadata = NunchakuFluxTransformer2dModel.from_pretrained(
                model_path,
                offload=cpu_offload_enabled,
                device=device,
                torch_dtype=torch.float16 if data_type == "float16" else torch.bfloat16,
                return_metadata=True,
            )
            self.model_path = model_path
            self.device = device
            self.cpu_offload = cpu_offload_enabled
            self.data_type = data_type
        self.transformer = apply_cache_on_transformer(
            transformer=self.transformer, residual_diff_threshold=cache_threshold
        )
        transformer = self.transformer
        if attention == "nunchaku-fp16":
            transformer.set_attention_impl("nunchaku-fp16")
        else:
            assert attention == "flash-attention2"
            transformer.set_attention_impl("flashattn2")

        if self.metadata is None:
            if os.path.exists(os.path.join(model_path, "comfy_config.json")):
                config_path = os.path.join(model_path, "comfy_config.json")
            else:
                default_config_root = os.path.join(os.path.dirname(__file__), "configs")
                config_name = os.path.basename(model_path).replace("svdq-int4-", "").replace("svdq-fp4-", "")
                config_path = os.path.join(default_config_root, f"{config_name}.json")
                assert os.path.exists(config_path), f"Config file not found: {config_path}"

            logger.info(f"Loading ComfyUI model config from {config_path}")
            comfy_config = json.load(open(config_path, "r"))
        else:
            comfy_config_str = self.metadata.get("comfy_config", None)
            comfy_config = json.loads(comfy_config_str)
        model_class_name = comfy_config["model_class"]

        model_config = comfy_config["model_config"]
        if "disable_unet_model_creation" not in model_config:
            model_config["disable_unet_model_creation"] = True

        if model_class_name == "FluxSchnell":
            model_class = FluxSchnell
        else:
            assert model_class_name == "Flux", f"Unknown model class {model_class_name}."
            model_class = Flux
        model_config = model_class(comfy_config["model_config"])
        model_config.set_inference_dtype(torch.bfloat16, None)
        model_config.custom_operations = None
        model = model_config.get_model({})
        model.diffusion_model = ComfyFluxWrapper(
            transformer,
            config=comfy_config["model_config"],
            ctx_for_copy={
                "comfy_config": comfy_config,
                "model_config": model_config,
                "device": device,
                "device_id": device_id,
            },
        )
        model = comfy.model_patcher.ModelPatcher(model, device, device_id)
        return (model,)
