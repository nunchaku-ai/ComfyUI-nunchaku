"""
This module provides a wrapper for the Z-Image transformer model (Lumina2/NextDiT),
enabling integration with ComfyUI forward, LoRA composition, and advanced caching strategies.

The LoRA application technique is adapted from ComfyUI-QwenImageLoraLoader:
https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader

Key structural differences between original Z-Image and Nunchaku-patched version:
- Original: layers.X.attention.to_q, to_k, to_v (separate)
- Nunchaku: layers.X.attention.qkv (fused)
- Original: layers.X.feed_forward.w1, w3 (separate)
- Nunchaku: layers.X.feed_forward.w13 (fused, output is [w3, w1])
- Original: layers.X.feed_forward.w2
- Nunchaku: layers.X.feed_forward.w2
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from comfy.model_patcher import ModelPatcher
from torch import nn

from nunchaku.lora.flux.nunchaku_converter import (
    pack_lowrank_weight,
    unpack_lowrank_weight,
)
from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.utils import load_state_dict_in_safetensors

logger = logging.getLogger(__name__)


# Regexes to extract LoRA suffix / alpha
_RE_LORA_SUFFIX = re.compile(r"\.lora_([AB])\.weight$")
_RE_ALPHA_SUFFIX = re.compile(r"\.(?:alpha|lora_alpha)(?:\.[^.]+)*$")

# Patterns to identify QKV components in LoRA keys
_RE_QKV_COMPONENT = re.compile(
    r"^(?:diffusion_model\.)?(?P<prefix>layers|context_refiner|noise_refiner)\.(?P<idx>\d+)\.attention\.to_(?P<comp>q|k|v)$"
)

# Pattern for attention output
_RE_ATTN_OUT = re.compile(
    r"^(?:diffusion_model\.)?(?P<prefix>layers|context_refiner|noise_refiner)\.(?P<idx>\d+)\.attention\.to_out\.0$"
)

# Patterns for feed-forward
_RE_FF_W1 = re.compile(
    r"^(?:diffusion_model\.)?(?P<prefix>layers|context_refiner|noise_refiner)\.(?P<idx>\d+)\.feed_forward\.w1$"
)
_RE_FF_W2 = re.compile(
    r"^(?:diffusion_model\.)?(?P<prefix>layers|context_refiner|noise_refiner)\.(?P<idx>\d+)\.feed_forward\.w2$"
)
_RE_FF_W3 = re.compile(
    r"^(?:diffusion_model\.)?(?P<prefix>layers|context_refiner|noise_refiner)\.(?P<idx>\d+)\.feed_forward\.w3$"
)

# Pattern for adaLN modulation
_RE_ADALN = re.compile(
    r"^(?:diffusion_model\.)?(?P<prefix>layers|context_refiner|noise_refiner)\.(?P<idx>\d+)\.adaLN_modulation\.0$"
)


class _LoRALinear(nn.Module):
    """
    Internal wrapper to apply LoRA updates to a standard nn.Linear layer
    during the forward pass without modifying the original weights permanently.
    """

    _LORA_A_BUF = "_nunchaku_lora_A"
    _LORA_B_BUF = "_nunchaku_lora_B"

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @staticmethod
    def _register_or_set_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
        if name in module._buffers:
            module._buffers[name] = tensor
            return
        # `persistent=False` avoids polluting state_dict; fallback for older torch.
        try:
            module.register_buffer(name, tensor, persistent=False)
        except TypeError:
            module.register_buffer(name, tensor)

    def set_loras(self, loras: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        if not loras:
            self.clear_loras()
            return

        A_cat = torch.cat([A for A, _ in loras], dim=0)
        B_cat = torch.cat([B for _, B in loras], dim=1)

        device = self.base.weight.device
        dtype = self.base.weight.dtype
        A_cat = A_cat.to(device=device, dtype=dtype)
        B_cat = B_cat.to(device=device, dtype=dtype)

        # Store as buffers on the leaf `nn.Linear` so ComfyUI's ModelPatcher `.to()` calls move them in lowvram mode.
        self._register_or_set_buffer(self.base, self._LORA_A_BUF, A_cat)
        self._register_or_set_buffer(self.base, self._LORA_B_BUF, B_cat)

    def clear_loras(self) -> None:
        self.base._buffers.pop(self._LORA_A_BUF, None)
        self.base._buffers.pop(self._LORA_B_BUF, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)

        A = self.base._buffers.get(self._LORA_A_BUF, None)
        B = self.base._buffers.get(self._LORA_B_BUF, None)
        if A is None or B is None:
            return out

        if A.device != x.device or A.dtype != x.dtype:
            A = A.to(device=x.device, dtype=x.dtype)
            B = B.to(device=x.device, dtype=x.dtype)

        return out + (x @ A.transpose(0, 1)) @ B.transpose(0, 1)


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Traverse a path like 'a.b.3.c' to find and return a module."""
    if not name:
        return model
    module = model
    for part in name.split("."):
        if not part:
            continue
        if hasattr(module, part):
            module = getattr(module, part)
        elif part.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
            try:
                module = module[int(part)]
            except (IndexError, TypeError):
                return None
        else:
            return None
    return module


def _block_diag(blocks: List[torch.Tensor]) -> torch.Tensor:
    """Construct a block diagonal matrix from a list of tensors."""
    out_total = sum(int(b.shape[0]) for b in blocks)
    r_total = sum(int(b.shape[1]) for b in blocks)
    if out_total == 0 or r_total == 0:
        raise ValueError("Empty blocks for block diagonal")

    dtype = blocks[0].dtype
    device = blocks[0].device
    out = torch.zeros((out_total, r_total), dtype=dtype, device=device)

    ro = 0
    co = 0
    for b in blocks:
        o, r = b.shape
        out[ro : ro + o, co : co + r] = b
        ro += o
        co += r
    return out


def _fuse_qkv_lora(
    q_A: torch.Tensor,
    q_B: torch.Tensor,
    k_A: torch.Tensor,
    k_B: torch.Tensor,
    v_A: torch.Tensor,
    v_B: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fuse separate Q, K, V LoRA weights into one fused QKV LoRA for a stacked QKV weight [q;k;v].

    If ranks differ between Q/K/V, the fused rank is sum of ranks, using a block-diagonal B.
    """
    if q_A.ndim != 2 or k_A.ndim != 2 or v_A.ndim != 2 or q_B.ndim != 2 or k_B.ndim != 2 or v_B.ndim != 2:
        logger.warning("Q/K/V A/B must be 2D")
        return None, None

    if not (q_A.shape[1] == k_A.shape[1] == v_A.shape[1]):
        logger.warning(f"Q/K/V in_features mismatch: Q={q_A.shape}, K={k_A.shape}, V={v_A.shape}")
        return None, None

    if not (q_A.shape[0] == q_B.shape[1] and k_A.shape[0] == k_B.shape[1] and v_A.shape[0] == v_B.shape[1]):
        logger.warning(
            f"Q/K/V rank mismatch: Q(A,B)={(q_A.shape, q_B.shape)}, K={(k_A.shape, k_B.shape)}, V={(v_A.shape, v_B.shape)}"
        )
        return None, None

    A_fused = torch.cat([q_A, k_A, v_A], dim=0)
    B_fused = _block_diag([q_B, k_B, v_B])
    return A_fused, B_fused


def _fuse_w13_lora(
    w3_A: torch.Tensor,
    w3_B: torch.Tensor,
    w1_A: torch.Tensor,
    w1_B: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fuse separate w1/w3 LoRAs into one LoRA for fused w13 where output is ordered [w3, w1].

    If ranks differ between w1 and w3, fused rank is sum of ranks via a block-diagonal B.
    """
    if w1_A.ndim != 2 or w3_A.ndim != 2 or w1_B.ndim != 2 or w3_B.ndim != 2:
        logger.warning("w1/w3 A/B must be 2D")
        return None, None

    if w1_A.shape[1] != w3_A.shape[1]:
        logger.warning(f"w1/w3 in_features mismatch: w1={w1_A.shape}, w3={w3_A.shape}")
        return None, None

    if w1_A.shape[0] != w1_B.shape[1] or w3_A.shape[0] != w3_B.shape[1]:
        logger.warning(f"w1/w3 rank mismatch: w1(A,B)={(w1_A.shape, w1_B.shape)} w3(A,B)={(w3_A.shape, w3_B.shape)}")
        return None, None

    # CRITICAL: w13 output is [w3, w1] to match Nunchaku's gating order.
    A_fused = torch.cat([w3_A, w1_A], dim=0)
    B_fused = _block_diag([w3_B, w1_B])
    return A_fused, B_fused


def _apply_lora_to_module(
    module: nn.Module, A: torch.Tensor, B: torch.Tensor, module_name: str, model: nn.Module
) -> bool:
    """
    Apply LoRA weights to a SVDQW4A4Linear module using concatenation technique.
    """
    if not isinstance(module, SVDQW4A4Linear):
        return False

    if A.ndim != 2 or B.ndim != 2:
        logger.warning(f"{module_name}: A/B must be 2D, got A={A.shape}, B={B.shape}")
        return False

    # LoRA A is typically [rank, in_features], B is [out_features, rank]
    # Verify dimensions match module
    if A.shape[1] != module.in_features:
        logger.warning(f"{module_name}: A in_features mismatch: A={A.shape}, module.in_features={module.in_features}")
        return False
    if B.shape[0] != module.out_features:
        logger.warning(
            f"{module_name}: B out_features mismatch: B={B.shape}, module.out_features={module.out_features}"
        )
        return False
    if A.shape[0] != B.shape[1]:
        logger.warning(f"{module_name}: A/B rank mismatch: A={A.shape}, B={B.shape}")
        return False

    # Get current proj_down and proj_up, unpacking from Nunchaku format
    pd = module.proj_down.data
    pu = module.proj_up.data

    pd = unpack_lowrank_weight(pd, down=True)
    pu = unpack_lowrank_weight(pu, down=False)

    # Determine the orientation of proj_down
    if pd.shape[1] == module.in_features:
        base_rank = pd.shape[0]
        # Cast LoRA vectors to the same device/dtype as weight base factors
        A_cast = A.to(pd.device, pd.dtype)
        B_cast = B.to(pu.device, pu.dtype)

        new_proj_down = torch.cat([pd, A_cast], dim=0)
        axis_down = 0
    else:
        base_rank = pd.shape[1]
        A_cast = A.to(pd.device, pd.dtype)
        B_cast = B.to(pu.device, pu.dtype)

        new_proj_down = torch.cat([pd, A_cast.T], dim=1)
        axis_down = 1

    new_proj_up = torch.cat([pu, B_cast], dim=1)

    # Pack back to Nunchaku format and update module
    module.proj_down.data = pack_lowrank_weight(new_proj_down, down=True)
    module.proj_up.data = pack_lowrank_weight(new_proj_up, down=False)
    module.rank = base_rank + A.shape[0]

    # Track LoRA slots on the model for reset capability
    if not hasattr(model, "_lora_slots"):
        model._lora_slots = {}
    model._lora_slots.setdefault(module_name, {"base_rank": base_rank, "axis_down": axis_down})

    return True


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> bool:
    parts = [p for p in name.split(".") if p]
    if not parts:
        return False

    parent_name = ".".join(parts[:-1])
    leaf = parts[-1]
    parent = _get_module_by_name(model, parent_name) if parent_name else model
    if parent is None:
        return False

    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = new_module
        return True

    setattr(parent, leaf, new_module)
    return True


def _apply_lora_to_linear(model: nn.Module, module_name: str, loras: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
    module = _get_module_by_name(model, module_name)
    if module is None:
        return False

    if isinstance(module, _LoRALinear):
        wrapper = module
    elif isinstance(module, nn.Linear):
        if not hasattr(model, "_lora_linear_slots"):
            model._lora_linear_slots = {}
        # Store the original module so reset_lora can restore types.
        model._lora_linear_slots.setdefault(module_name, module)
        wrapper = _LoRALinear(module)
        if not _set_module_by_name(model, module_name, wrapper):
            return False
    else:
        return False

    wrapper.set_loras(loras)

    return True


def reset_lora(model: nn.Module) -> None:
    """Reset all LoRA modifications, restoring modules to their base rank."""
    # Restore any wrapped nn.Linear modules (adaLN_modulation.0) back to their original nn.Linear.
    lora_linear_slots = getattr(model, "_lora_linear_slots", None)
    if lora_linear_slots:
        for module_name, orig_linear in list(lora_linear_slots.items()):
            # Ensure any attached LoRA buffers are cleared before restoring.
            orig_linear._buffers.pop(_LoRALinear._LORA_A_BUF, None)
            orig_linear._buffers.pop(_LoRALinear._LORA_B_BUF, None)
            current = _get_module_by_name(model, module_name)
            if isinstance(current, _LoRALinear):
                _set_module_by_name(model, module_name, orig_linear)
        lora_linear_slots.clear()

    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        return

    for module_name, slot_info in model._lora_slots.items():
        module = _get_module_by_name(model, module_name)
        if module is None or not isinstance(module, SVDQW4A4Linear):
            continue

        base_rank = slot_info["base_rank"]
        axis_down = slot_info["axis_down"]

        pd = unpack_lowrank_weight(module.proj_down.data, down=True)
        pu = unpack_lowrank_weight(module.proj_up.data, down=False)

        if axis_down == 0:
            pd = pd[:base_rank, :]
        else:
            pd = pd[:, :base_rank]
        pu = pu[:, :base_rank]

        module.proj_down.data = pack_lowrank_weight(pd, down=True)
        module.proj_up.data = pack_lowrank_weight(pu, down=False)
        module.rank = base_rank

    model._lora_slots.clear()
    logger.debug("[Z-Image LoRA] Reset all LoRA modifications")


def compose_loras(model: nn.Module, lora_configs: List[Tuple[Union[str, Path], float]]) -> int:
    """
    Compose multiple LoRAs into the Z-Image model with proper Q/K/V fusion.

    Handles the structural difference between original Z-Image LoRAs (separate Q/K/V)
    and Nunchaku's fused QKV layers.
    """
    reset_lora(model)

    if not lora_configs:
        return 0

    # Build SVD module name -> module mapping for quick lookup
    svdq_map: Dict[str, SVDQW4A4Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, SVDQW4A4Linear):
            svdq_map[name] = module

    logger.debug(f"[Z-Image LoRA] Found {len(svdq_map)} SVDQW4A4Linear modules")

    # Aggregate updates by module as a list of independent low-rank factors.
    svdq_updates: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    linear_updates: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)

    for lora_path, strength in lora_configs:
        lora_sd = load_state_dict_in_safetensors(str(lora_path))
        logger.info(f"[Z-Image LoRA] Loading {Path(lora_path).name} (strength={strength:.2f})")

        # Collect per-module alpha values (optional).
        alpha_map: Dict[str, float] = {}
        for key, value in lora_sd.items():
            m_alpha = _RE_ALPHA_SUFFIX.search(key)
            if not m_alpha:
                continue
            base_key = key[: m_alpha.start()]
            try:
                alpha_map[base_key] = float(value.item())
            except Exception:
                continue

        qkv_parts: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = defaultdict(dict)
        w13_parts: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = defaultdict(dict)
        regular_parts: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        adaln_parts: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        unmatched = 0

        for key, tensor in lora_sd.items():
            m = _RE_LORA_SUFFIX.search(key)
            if not m:
                continue
            ab = m.group(1)  # A or B
            base_key = key[: m.start()]

            # Scale is applied to B only: scale = strength * (alpha / rank).
            # If alpha is missing, treat alpha == rank (scale == strength).
            scale = strength
            # NOTE: alpha keys (if present) are stored on the same base_key.
            if ab == "B":
                # We need rank to compute alpha/rank; rank is A.shape[0] but A may not be present yet.
                # Use B.shape[1] as rank for standard LoRA: B is [out, rank].
                rank = int(tensor.shape[1]) if tensor.ndim == 2 else None
                alpha = alpha_map.get(base_key, None)
                if alpha is not None and rank is not None and rank > 0:
                    scale = strength * (alpha / rank)
                tensor = tensor * scale

            m_qkv = _RE_QKV_COMPONENT.match(base_key)
            if m_qkv:
                prefix = m_qkv.group("prefix")
                idx = m_qkv.group("idx")
                comp = m_qkv.group("comp")  # q, k, v
                qkv_parts[(prefix, idx)][f"{comp}_{ab}"] = tensor
                continue

            m_out = _RE_ATTN_OUT.match(base_key)
            if m_out:
                prefix = m_out.group("prefix")
                idx = m_out.group("idx")
                target = f"{prefix}.{idx}.attention.out"
                regular_parts[target][ab] = tensor
                continue

            m_w2 = _RE_FF_W2.match(base_key)
            if m_w2:
                prefix = m_w2.group("prefix")
                idx = m_w2.group("idx")
                target = f"{prefix}.{idx}.feed_forward.w2"
                regular_parts[target][ab] = tensor
                continue

            m_w1 = _RE_FF_W1.match(base_key)
            if m_w1:
                prefix = m_w1.group("prefix")
                idx = m_w1.group("idx")
                w13_parts[(prefix, idx)][f"w1_{ab}"] = tensor
                continue

            m_w3 = _RE_FF_W3.match(base_key)
            if m_w3:
                prefix = m_w3.group("prefix")
                idx = m_w3.group("idx")
                w13_parts[(prefix, idx)][f"w3_{ab}"] = tensor
                continue

            m_adaln = _RE_ADALN.match(base_key)
            if m_adaln:
                prefix = m_adaln.group("prefix")
                idx = m_adaln.group("idx")
                target = f"{prefix}.{idx}.adaLN_modulation.0"
                adaln_parts[target][ab] = tensor
                continue

            unmatched += 1

        # Fuse QKV per layer for this LoRA.
        fused_qkv = 0
        for (prefix, idx), parts in qkv_parts.items():
            needed = ["q_A", "q_B", "k_A", "k_B", "v_A", "v_B"]
            if not all(k in parts for k in needed):
                logger.warning(f"[Z-Image LoRA] Incomplete Q/K/V for {prefix}.{idx}: has {sorted(parts.keys())}")
                continue
            A_fused, B_fused = _fuse_qkv_lora(
                parts["q_A"],
                parts["q_B"],
                parts["k_A"],
                parts["k_B"],
                parts["v_A"],
                parts["v_B"],
            )
            if A_fused is None or B_fused is None:
                continue
            target = f"{prefix}.{idx}.attention.qkv"
            svdq_updates[target].append((A_fused, B_fused))
            fused_qkv += 1

        # Fuse w1/w3 into w13 per layer for this LoRA.
        fused_w13 = 0
        for (prefix, idx), parts in w13_parts.items():
            needed = ["w1_A", "w1_B", "w3_A", "w3_B"]
            if not all(k in parts for k in needed):
                logger.warning(f"[Z-Image LoRA] Incomplete w1/w3 for {prefix}.{idx}: has {sorted(parts.keys())}")
                continue
            A_fused, B_fused = _fuse_w13_lora(parts["w3_A"], parts["w3_B"], parts["w1_A"], parts["w1_B"])
            if A_fused is None or B_fused is None:
                continue
            target = f"{prefix}.{idx}.feed_forward.w13"
            svdq_updates[target].append((A_fused, B_fused))
            fused_w13 += 1

        # Regular SVDQW4A4Linear targets for this LoRA.
        regular = 0
        for target, parts in regular_parts.items():
            A = parts.get("A")
            B = parts.get("B")
            if A is None or B is None:
                continue
            svdq_updates[target].append((A, B))
            regular += 1

        # adaLN_modulation.0 (nn.Linear) targets for this LoRA.
        adaln = 0
        for target, parts in adaln_parts.items():
            A = parts.get("A")
            B = parts.get("B")
            if A is None or B is None:
                continue
            linear_updates[target].append((A, B))
            adaln += 1

        logger.debug(
            f"[Z-Image LoRA] Parsed {Path(lora_path).name}: fused_qkv={fused_qkv}, fused_w13={fused_w13}, regular={regular}, adaln={adaln}, unmatched={unmatched}"
        )

    # Apply aggregated SVDQ updates (concatenate ranks; do not sum A/B).
    applied = 0
    for module_path, parts in svdq_updates.items():
        module = svdq_map.get(module_path)
        if module is None:
            logger.debug(f"[Z-Image LoRA] SVDQ module not found: {module_path}")
            continue

        A_cat = torch.cat([a for a, _ in parts], dim=0)
        B_cat = torch.cat([b for _, b in parts], dim=1)

        if _apply_lora_to_module(module, A_cat, B_cat, module_path, model):
            applied += 1

    # Apply aggregated nn.Linear updates using a forward-time wrapper.
    for module_path, parts in linear_updates.items():
        if not parts:
            continue
        if _apply_lora_to_linear(model, module_path, parts):
            applied += 1
        else:
            logger.debug(f"[Z-Image LoRA] Linear module not found: {module_path}")

    if applied == 0:
        logger.warning("[Z-Image LoRA] Applied 0 modules; check module-path mapping and LoRA key formats")

    logger.info(f"[Z-Image LoRA] Applied LoRA to {applied} modules")
    return applied


class ComfyZImageWrapper(nn.Module):
    """
    Wrapper for Z-Image transformer to support ComfyUI workflows and LoRA composition.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        ctx_for_copy: Optional[dict] = None,
    ):
        super(ComfyZImageWrapper, self).__init__()
        self.model = model
        try:
            self.dtype = next(model.parameters()).dtype
        except StopIteration:
            self.dtype = torch.float16

        self.config = config
        self.loras: List[Tuple[Union[str, Path], float]] = []
        ctx_for_copy = {} if ctx_for_copy is None else ctx_for_copy
        self.ctx_for_copy = ctx_for_copy.copy()
        self._applied_loras: List[Tuple[Union[str, Path], float]] = []

    def forward(self, *args, **kwargs):
        """Forward pass - composes LoRAs if changed, then calls underlying model."""
        model = self.model

        if not self.loras and getattr(model, "_lora_slots", None):
            reset_lora(model)
            self._applied_loras = []

        if self.loras != self._applied_loras:
            compose_loras(model, self.loras)
            self._applied_loras = self.loras.copy()

        return model(*args, **kwargs)

    def to(self, *args, **kwargs):
        """Override to() to ensure safe offloading via to_safely check."""
        if hasattr(self.model, "to_safely"):
            self.model.to_safely(*args, **kwargs)
            return self
        return super().to(*args, **kwargs)

    def to_safely(self, *args, **kwargs):
        """Delegate to_safely to the underlying model to support ComfyUI offloading."""
        if hasattr(self.model, "to_safely"):
            self.model.to_safely(*args, **kwargs)
            return self
        self.model.to(*args, **kwargs)
        return self


def copy_with_ctx(model_wrapper: ComfyZImageWrapper) -> Tuple[ComfyZImageWrapper, ModelPatcher]:
    """Duplicates a ComfyZImageWrapper object with its initialization context."""
    ctx_for_copy = model_wrapper.ctx_for_copy
    ret_model_wrapper = ComfyZImageWrapper(
        model_wrapper.model,
        config=model_wrapper.config,
        ctx_for_copy=ctx_for_copy,
    )
    ret_model_wrapper.loras = model_wrapper.loras.copy()

    model_config = ctx_for_copy.get("model_config")
    if model_config is None:
        raise ValueError("model_config missing in ctx_for_copy")

    model_base = model_config.get_model({})
    model_base.diffusion_model = ret_model_wrapper

    device = ctx_for_copy.get("device", torch.device("cpu"))
    device_id = ctx_for_copy.get("device_id", 0)
    offload_device = ctx_for_copy.get("offload_device", torch.device("cpu"))

    ret_model = ModelPatcher(model_base, load_device=device, offload_device=offload_device)
    return ret_model_wrapper, ret_model
