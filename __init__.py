import logging
import os
from pathlib import Path

import torch
import yaml
from packaging.version import InvalidVersion, Version
# --- ComfyUI compatibility hotfix for Flux init (Fix #801-like breakage) ---
def _nunchaku_patch_comfy_flux_init():
    try:
        import torch
        import comfy.model_base as model_base
    except Exception:
        return

    def _patch(cls):
        if cls is None:
            return
        if getattr(cls, "_nunchaku_dm_hotfix", False):
            return

        orig_init = cls.__init__

        def patched_init(self, *args, __orig_init=orig_init, **kwargs):
            # IMPORTANT:
            # Do NOT use hasattr/getattr here (nn.Module isn't fully initialized yet).
            # Put a temporary instance attribute in __dict__ so BaseModel init won't crash.
            if "diffusion_model" not in self.__dict__:
                self.__dict__["diffusion_model"] = torch.nn.Identity()
            return __orig_init(self, *args, **kwargs)

        cls.__init__ = patched_init
        cls._nunchaku_dm_hotfix = True

    _patch(getattr(model_base, "Flux", None))
    _patch(getattr(model_base, "FluxSchnell", None))

_nunchaku_patch_comfy_flux_init()
# --- end hotfix ---

# vanilla and LTS compatibility snippet
try:
    from comfy_compatibility.vanilla import prepare_vanilla_environment

    prepare_vanilla_environment()

    from comfy.model_downloader import add_known_models
    from comfy.model_downloader_types import HuggingFile

    capability = torch.cuda.get_device_capability(0 if torch.cuda.is_available() else None)
    sm = f"{capability[0]}{capability[1]}"
    precision = "fp4" if sm == "120" else "int4"

    # add known models

    models_yaml_path = Path(__file__).parent / "test_data" / "models.yaml"
    with open(models_yaml_path, "r") as f:
        nunchaku_models_yaml = yaml.safe_load(f)

    NUNCHAKU_SVDQ_MODELS = []
    for model in nunchaku_models_yaml["models"]:
        filename = model["filename"]
        if not filename.startswith("svdq-"):
            continue
        if "{precision}" in filename:
            filename = filename.format(precision=precision)
        NUNCHAKU_SVDQ_MODELS.append(HuggingFile(repo_id=model["repo_id"], filename=filename))

    NUNCHAKU_SVDQ_TEXT_ENCODER_MODELS = [
        HuggingFile(repo_id="nunchaku-tech/nunchaku-t5", filename="awq-int4-flux.1-t5xxl.safetensors"),
    ]

    add_known_models("diffusion_models", *NUNCHAKU_SVDQ_MODELS)
    add_known_models("text_encoders", *NUNCHAKU_SVDQ_TEXT_ENCODER_MODELS)
except (ImportError, ModuleNotFoundError):
    pass

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-nunchaku Initialization " + "=" * 40)

from .utils import get_package_version, get_plugin_version

nunchaku_full_version = get_package_version("nunchaku").split("+")[0].strip()

logger.info(f"Nunchaku version: {nunchaku_full_version}")
logger.info(f"ComfyUI-nunchaku version: {get_plugin_version()}")


min_nunchaku_version = "1.0.0"
nunchaku_version = nunchaku_full_version.split("+")[0].strip()
nunchaku_major_minor_patch_version = ".".join(nunchaku_version.split(".")[:3])

try:
    if Version(nunchaku_major_minor_patch_version) < Version(min_nunchaku_version):
        logger.warning(
            f"ComfyUI-nunchaku {get_plugin_version()} requires nunchaku >= v{min_nunchaku_version}, "
            f"but found nunchaku {nunchaku_full_version}. Please update nunchaku."
        )
except InvalidVersion:
    logger.warning(
        f"Could not parse nunchaku version: {nunchaku_full_version}. "
        f"Please ensure you have at least v{min_nunchaku_version}."
    )

NODE_CLASS_MAPPINGS = {}

try:
    from .nodes.models.flux import NunchakuFluxDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"] = NunchakuFluxDiTLoader
except ImportError:
    logger.exception("Node `NunchakuFluxDiTLoader` import failed:")

try:
    from .nodes.models.qwenimage import NunchakuQwenImageDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuQwenImageDiTLoader"] = NunchakuQwenImageDiTLoader
except ImportError:
    logger.exception("Node `NunchakuQwenImageDiTLoader` import failed:")

try:
    from .nodes.lora.flux import NunchakuFluxLoraLoader, NunchakuFluxLoraStack

    NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"] = NunchakuFluxLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuFluxLoraStack"] = NunchakuFluxLoraStack
except ImportError:
    logger.exception("Nodes `NunchakuFluxLoraLoader` and `NunchakuFluxLoraStack` import failed:")

try:
    from .nodes.models.text_encoder import NunchakuTextEncoderLoader, NunchakuTextEncoderLoaderV2

    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoader"] = NunchakuTextEncoderLoader
    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoaderV2"] = NunchakuTextEncoderLoaderV2
except ImportError:
    logger.exception("Nodes `NunchakuTextEncoderLoader` and `NunchakuTextEncoderLoaderV2` import failed:")

try:
    from .nodes.preprocessors.depth import FluxDepthPreprocessor

    NODE_CLASS_MAPPINGS["NunchakuDepthPreprocessor"] = FluxDepthPreprocessor
except ImportError:
    logger.exception("Node `NunchakuDepthPreprocessor` import failed:")

try:
    from .nodes.models.pulid import (
        NunchakuFluxPuLIDApplyV2,
        NunchakuPulidApply,
        NunchakuPulidLoader,
        NunchakuPuLIDLoaderV2,
    )

    NODE_CLASS_MAPPINGS["NunchakuPulidApply"] = NunchakuPulidApply
    NODE_CLASS_MAPPINGS["NunchakuPulidLoader"] = NunchakuPulidLoader
    NODE_CLASS_MAPPINGS["NunchakuPuLIDLoaderV2"] = NunchakuPuLIDLoaderV2
    NODE_CLASS_MAPPINGS["NunchakuFluxPuLIDApplyV2"] = NunchakuFluxPuLIDApplyV2
except ImportError:
    logger.exception(
        "Nodes `NunchakuPulidApply`,`NunchakuPulidLoader`, "
        "`NunchakuPuLIDLoaderV2` and `NunchakuFluxPuLIDApplyV2` import failed:"
    )
try:
    from .nodes.models.ipadapter import NunchakuFluxIPAdapterApply, NunchakuIPAdapterLoader

    NODE_CLASS_MAPPINGS["NunchakuFluxIPAdapterApply"] = NunchakuFluxIPAdapterApply
    NODE_CLASS_MAPPINGS["NunchakuIPAdapterLoader"] = NunchakuIPAdapterLoader
except ImportError:
    logger.exception("Nodes `NunchakuFluxIPAdapterApply` and `NunchakuIPAdapterLoader` import failed:")

try:
    from .nodes.models.zimage import NunchakuZImageDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuZImageDiTLoader"] = NunchakuZImageDiTLoader
except ImportError:
    logger.exception("Nodes `NunchakuZImageDiTLoader` import failed:")

try:
    from .nodes.tools.merge_safetensors import NunchakuModelMerger

    NODE_CLASS_MAPPINGS["NunchakuModelMerger"] = NunchakuModelMerger
except ImportError:
    logger.exception("Node `NunchakuModelMerger` import failed:")

try:
    from .nodes.tools.installers import NunchakuWheelInstaller

    NODE_CLASS_MAPPINGS["NunchakuWheelInstaller"] = NunchakuWheelInstaller
except ImportError:
    logger.exception("Node `NunchakuWheelInstaller` import failed:")

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
logger.info("=" * (80 + len(" ComfyUI-nunchaku Initialization ")))
