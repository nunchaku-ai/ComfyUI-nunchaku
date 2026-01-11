import functools
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors import safe_open

# It's assumed these functions from your project are available for import.
# If not, you'll need to provide their definitions.
from nunchaku.lora.flux.nunchaku_converter import (
    pack_lowrank_weight,
    reorder_adanorm_lora_up,
    unpack_lowrank_weight,
)

logger = logging.getLogger(__name__)

# --- Centralized & Optimized Key Mapping ---
# This structure is faster to process and easier to maintain than a long if/elif chain.
# --- CORRECTED Centralized & Optimized Key Mapping ---
# --- Centralized & Optimized Key Mapping ---
# This version correctly tokenizes all module paths.
KEY_MAPPING = [
    # Fused QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]qkv$"), r"\1.\2.attn.to_qkv", "qkv", None),
    # Decomposed QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._](q|k|v)$"), r"\1.\2.attn.to_qkv", "qkv",
     lambda m: m.group(3).upper()),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._](q|k|v)[._]proj$"), r"\1.\2.attn.to_qkv", "qkv",
     lambda m: m.group(3).upper()),
    # Fused Add_QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]add[._]qkv[._]proj$"), r"\1.\2.attn.add_qkv_proj",
     "add_qkv", None),
    # Decomposed Add_QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]add[._](q|k|v)[._]proj$"), r"\1.\2.attn.add_qkv_proj",
     "add_qkv",
     lambda m: m.group(3).upper()),
    # Fused QKV (Single Block)
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._]qkv$"), r"\1.\2.attn.to_qkv", "qkv", None),
    # Decomposed QKV (Single Block)
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._](q|k|v)$"), r"\1.\2.attn.to_qkv", "qkv",
     lambda m: m.group(3).upper()),
    # Output Projections
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]out[._]proj[._]context$"), r"\1.\2.attn.to_add_out", "regular",
     None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]add[._]out$"), r"\1.\2.attn.to_add_out",
     "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]out[._]proj$"), r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]out$"), r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]out[._]0$"), r"\1.\2.attn.to_out.0", "regular",
     None),
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._]out$"), r"\1.\2.attn.to_out", "regular",
     None),

    # Feed-Forward / MLP Layers (Standard)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff[._]net[._]0(?:[._]proj)?$"), r"\1.\2.mlp_fc1", "regular",
     None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff[._]net[._]2$"), r"\1.\2.mlp_fc2", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff_context[._]net[._]0(?:[._]proj)?$"),
     r"\1.\2.mlp_context_fc1", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff_context[._]net[._]2$"), r"\1.\2.mlp_context_fc2", "regular",
     None),

    # --- THIS IS THE CORRECTED SECTION ---
    # Feed-Forward / MLP Layers (img/txt)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](0)[._](proj)$"), r"\1.\2.\3.\4.\5.\6",
     "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](0)$"), r"\1.\2.\3.\4.\5", "regular",
     None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](2)$"), r"\1.\2.\3.\4.\5", "regular",
     None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](0)[._](proj)$"), r"\1.\2.\3.\4.\5.\6",
     "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](0)$"), r"\1.\2.\3.\4.\5", "regular",
     None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](2)$"), r"\1.\2.\3.\4.\5", "regular",
     None),
    # Mod Layers (img/txt)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mod)[._](1)$"), r"\1.\2.\3.\4", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mod)[._](1)$"), r"\1.\2.\3.\4", "regular", None),
    # ------------------------------------

    # Single Block Projections
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]proj[._]out$"), r"\1.\2.proj_out", "single_proj_out",
     None),
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]proj[._]mlp$"), r"\1.\2.mlp_fc1", "regular", None),
    # Normalization Layers
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]norm[._]linear$"), r"\1.\2.norm.linear", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]norm1[._]linear$"), r"\1.\2.norm1.linear", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]norm1_context[._]linear$"), r"\1.\2.norm1_context.linear",
     "regular", None),

    # Mappings for top-level diffusion_model modules
    (re.compile(r"^(img_in)$"), r"\1", "regular", None),
    (re.compile(r"^(txt_in)$"), r"\1", "regular", None),
    (re.compile(r"^(proj_out)$"), r"\1", "regular", None),
    (re.compile(r"^(norm_out)[._](linear)$"), r"\1.\2", "regular", None),
    (re.compile(r"^(time_text_embed)[._](timestep_embedder)[._](linear_1)$"), r"\1.\2.\3", "regular", None),
    (re.compile(r"^(time_text_embed)[._](timestep_embedder)[._](linear_2)$"), r"\1.\2.\3", "regular", None),
]
_RE_LORA_SUFFIX = re.compile(r"\.(?P<tag>lora(?:[._](?:A|B|down|up)))(?:\.[^.]+)*\.weight$")
_RE_ALPHA_SUFFIX = re.compile(r"\.(?:alpha|lora_alpha)(?:\.[^.]+)*$")


# --- Helper Functions ---
def _rename_layer_underscore_layer_name(old_name: str) -> str:
    """
    Converts specific model layer names by replacing underscore patterns
    with dot notation using an ordered set of regex rules.
    """

    # Rules are ordered from most specific to most general
    # to prevent a general rule from incorrectly matching part
    # of a more specific pattern.
    rules = [
        # Case: transformer_blocks_8_attn_to_out_0 -> transformer_blocks.8.attn.to_out.0
        (r'_(\d+)_attn_to_out_(\d+)', r'.\1.attn.to_out.\2'),

        # Case: transformer_blocks_8_img_mlp_net_0_proj -> transformer_blocks.8.img_mlp.net.0.proj
        (r'_(\d+)_img_mlp_net_(\d+)_proj', r'.\1.img_mlp.net.\2.proj'),

        # Case: transformer_blocks_8_txt_mlp_net_0_proj -> transformer_blocks.8.txt_mlp.net.0.proj
        (r'_(\d+)_txt_mlp_net_(\d+)_proj', r'.\1.txt_mlp.net.\2.proj'),

        # Case: transformer_blocks_8_img_mlp_net_2 -> transformer_blocks.8.img_mlp.net.2
        (r'_(\d+)_img_mlp_net_(\d+)', r'.\1.img_mlp.net.\2'),

        # Case: transformer_blocks_8_txt_mlp_net_2 -> transformer_blocks.8.txt_mlp.net.2
        (r'_(\d+)_txt_mlp_net_(\d+)', r'.\1.txt_mlp.net.\2'),

        # Case: transformer_blocks_8_img_mod_1 -> transformer_blocks.8.img_mod.1
        (r'_(\d+)_img_mod_(\d+)', r'.\1.img_mod.\2'),

        # Case: transformer_blocks_8_txt_mod_1 -> transformer_blocks.8.txt_mod.1
        (r'_(\d+)_txt_mod_(\d+)', r'.\1.txt_mod.\2'),

        # General 'attn' case: transformer_blocks_8_attn_... -> transformer_blocks.8.attn....
        # This catches add_k_proj, add_q_proj, to_k, etc.
        (r'_(\d+)_attn_', r'.\1.attn.'),
    ]

    new_name = old_name
    for pattern, replacement in rules:
        # Apply the substitution. If the pattern doesn't match,
        # re.sub simply returns the original string.
        new_name = re.sub(pattern, replacement, new_name)

    return new_name

def _classify_and_map_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    """
    Efficiently classifies a LoRA key using the centralized KEY_MAPPING.
    The implementation is new and optimized, but the name and signature are preserved.
    """
    k = key
    if k.startswith("transformer."):
        k = k[len("transformer."):]
    if k.startswith("diffusion_model."):
        k = k[len("diffusion_model."):]
    if k.startswith("lora_unet_"):
        k = k[len("lora_unet_"):]
        k = _rename_layer_underscore_layer_name(k)


    base = None
    ab = None

    m = _RE_LORA_SUFFIX.search(k)
    if m:
        tag = m.group("tag")
        base = k[: m.start()]
        if "lora_A" in tag or tag.endswith(".A") or "down" in tag:
            ab = "A"
        elif "lora_B" in tag or tag.endswith(".B") or "up" in tag:
            ab = "B"
    else:
        m = _RE_ALPHA_SUFFIX.search(k)
        if m:
            ab = "alpha"
            base = k[: m.start()]

    if base is None or ab is None:
        return None  # Not a recognized LoRA key format

    for pattern, template, group, comp_fn in KEY_MAPPING:
        match = pattern.match(base)
        if match:
            final_key = match.expand(template)
            component = comp_fn(match) if comp_fn else None
            return group, final_key, component, ab

    return None


def _is_indexable_module(m):
    """Checks if a module is a list-like container."""
    return isinstance(m, (nn.ModuleList, nn.Sequential, list, tuple))


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Traverse a path like 'a.b.3.c' to find and return a module."""
    if not name: return model
    module = model
    for part in name.split("."):
        if not part: continue

        # Prioritize hasattr check. This works for:
        # 1. Regular attributes ('attn', 'img_mod')
        # 2. Numerically-named children in nn.Sequential/nn.ModuleDict ('0', '1', '2')
        if hasattr(module, part):
            module = getattr(module, part)
        # Fallback to indexing for ModuleList (which fails hasattr for numeric keys)
        elif part.isdigit() and _is_indexable_module(module):
            try:
                module = module[int(part)]
            except (IndexError, TypeError):
                logger.warning(f"Failed to index module {name} with part {part}")
                return None
        # All attempts failed
        else:
            return None
    return module

def _resolve_module_name(model: nn.Module, name: str) -> Tuple[str, Optional[nn.Module]]:
    """Resolve a name string path to a module, attempting fallback paths."""
    m = _get_module_by_name(model, name)
    if m is not None:
        return name, m

    if name.endswith(".attn.to_out.0"):
        alt = name[:-2]
        m = _get_module_by_name(model, alt)
        if m is not None: return alt, m
    elif name.endswith(".attn.to_out"):
        alt = name + ".0"
        m = _get_module_by_name(model, alt)
        if m is not None: return alt, m

    mapping = {
        ".ff.net.0.proj": ".mlp_fc1", ".ff.net.2": ".mlp_fc2",
        ".ff_context.net.0.proj": ".mlp_context_fc1", ".ff_context.net.2": ".mlp_context_fc2",
    }
    for src, dst in mapping.items():
        if src in name:
            alt = name.replace(src, dst)
            m = _get_module_by_name(model, alt)
            if m is not None: return alt, m

    logger.debug(f"[MISS] Module not found: {name}")
    return name, None


def _load_lora_state_dict(lora_state_dict_or_path: Union[str, Path, Dict[str, torch.Tensor]]) -> Dict[
    str, torch.Tensor]:
    """Load LoRA state dict from path or return existing dict."""
    if isinstance(lora_state_dict_or_path, (str, Path)):
        path = Path(lora_state_dict_or_path)
        if path.suffix == ".safetensors":
            state_dict = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            return state_dict
        else:
            return torch.load(path, map_location="cpu")
    return lora_state_dict_or_path


def _fuse_qkv_lora(qkv_weights: Dict[str, torch.Tensor]) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse Q/K/V LoRA weights into a single QKV tensor."""
    required_keys = ["Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B"]
    if not all(k in qkv_weights for k in required_keys):
        return None, None, None

    A_q, A_k, A_v = qkv_weights["Q_A"], qkv_weights["K_A"], qkv_weights["V_A"]
    B_q, B_k, B_v = qkv_weights["Q_B"], qkv_weights["K_B"], qkv_weights["V_B"]

    if not (A_q.shape == A_k.shape == A_v.shape):
        logger.warning(f"Q/K/V LoRA A dimensions mismatch: {A_q.shape}, {A_k.shape}, {A_v.shape}")
        return None, None, None

    alpha_q, alpha_k, alpha_v = qkv_weights.get("Q_alpha"), qkv_weights.get("K_alpha"), qkv_weights.get("V_alpha")
    alpha_fused = None
    if alpha_q is not None and alpha_k is not None and alpha_v is not None and (
            alpha_q.item() == alpha_k.item() == alpha_v.item()):
        alpha_fused = alpha_q

    A_fused = torch.cat([A_q, A_k, A_v], dim=0)

    r = B_q.shape[1]
    out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
    B_fused = torch.zeros(out_q + out_k + out_v, 3 * r, dtype=B_q.dtype, device=B_q.device)
    B_fused[:out_q, :r] = B_q
    B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
    B_fused[out_q + out_k:, 2 * r:] = B_v

    return A_fused, B_fused, alpha_fused


def _handle_proj_out_split(lora_dict: Dict[str, Dict[str, torch.Tensor]], base_key: str, model: nn.Module) -> Tuple[
    Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]], List[str]]:
    """Split single-block proj_out LoRA into two branches."""
    result, consumed = {}, []
    m = re.search(r"single_transformer_blocks\.(\d+)", base_key)
    if not m or base_key not in lora_dict:
        return result, consumed

    block_idx = m.group(1)
    block = _get_module_by_name(model, f"single_transformer_blocks.{block_idx}")
    if block is None:
        return result, consumed

    A_full, B_full, alpha = lora_dict[base_key].get("A"), lora_dict[base_key].get("B"), lora_dict[base_key].get("alpha")
    if A_full is None or B_full is None:
        return result, consumed

    attn_to_out = getattr(getattr(block, "attn", None), "to_out", None)
    mlp_fc2 = getattr(block, "mlp_fc2", None)
    if attn_to_out is None or mlp_fc2 is None or not hasattr(attn_to_out, "in_features") or not hasattr(mlp_fc2,
                                                                                                        "in_features"):
        return result, consumed

    attn_in, mlp_in = attn_to_out.in_features, mlp_fc2.in_features
    if A_full.shape[1] != attn_in + mlp_in:
        logger.warning(f"{base_key}: A_full shape mismatch {A_full.shape} vs expected in_features {attn_in + mlp_in}")
        return result, consumed

    A_attn, A_mlp = A_full[:, :attn_in], A_full[:, attn_in:]
    result[f"single_transformer_blocks.{block_idx}.attn.to_out"] = (A_attn, B_full.clone(), alpha)
    result[f"single_transformer_blocks.{block_idx}.mlp_fc2"] = (A_mlp, B_full.clone(), alpha)
    consumed.append(base_key)
    return result, consumed


def _apply_lora_to_module(module: nn.Module, A: torch.Tensor, B: torch.Tensor, module_name: str,
                          model: nn.Module) -> None:
    """Helper to append combined LoRA weights to a module."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{module_name}: A/B must be 2D, got {A.shape}, {B.shape}")
    if A.shape[1] != module.in_features:
        raise ValueError(f"{module_name}: A shape {A.shape} mismatch with in_features={module.in_features}")
    if B.shape[0] != module.out_features:
        raise ValueError(f"{module_name}: B shape {B.shape} mismatch with out_features={module.out_features}")

    pd, pu = module.proj_down.data, module.proj_up.data
    pd = unpack_lowrank_weight(pd, down=True)
    pu = unpack_lowrank_weight(pu, down=False)

    base_rank = pd.shape[0] if pd.shape[1] == module.in_features else pd.shape[1]

    if pd.shape[1] == module.in_features:  # [rank, in]
        new_proj_down = torch.cat([pd, A], dim=0)
        axis_down = 0
    else:  # [in, rank]
        new_proj_down = torch.cat([pd, A.T], dim=1)
        axis_down = 1

    new_proj_up = torch.cat([pu, B], dim=1)

    module.proj_down.data = pack_lowrank_weight(new_proj_down, down=True)
    module.proj_up.data = pack_lowrank_weight(new_proj_up, down=False)
    module.rank = base_rank + A.shape[0]

    if not hasattr(model, "_lora_slots"):
        model._lora_slots = {}
    slot = model._lora_slots.setdefault(module_name, {"base_rank": base_rank, "appended": 0, "axis_down": axis_down})
    slot["appended"] += A.shape[0]


# --- Main Public API ---

def compose_loras_v2(
        model: torch.nn.Module,
        lora_configs: List[Tuple[Union[str, Path, Dict[str, torch.Tensor]], float]],
) -> None:
    """
    Resets and composes multiple LoRAs into the model with individual strengths.
    """
    logger.info(f"Composing {len(lora_configs)} LoRAs...")
    reset_lora_v2(model)

    aggregated_weights: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # 1. Aggregate weights from all LoRAs
    for lora_path_or_dict, strength in lora_configs:
        lora_name = lora_path_or_dict if isinstance(lora_path_or_dict, str) else "dict"
        lora_state_dict = _load_lora_state_dict(lora_path_or_dict)

        lora_grouped: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        for key, value in lora_state_dict.items():
            parsed = _classify_and_map_key(key)
            if parsed is None:
                continue

            group, base_key, comp, ab = parsed
            if group in ("qkv", "add_qkv") and comp is not None:
                lora_grouped[base_key][f"{comp}_{ab}"] = value
            else:
                lora_grouped[base_key][ab] = value

        # Process grouped weights for this LoRA
        processed_groups = {}
        special_handled = set()
        for base_key, lw in lora_grouped.items():
            if base_key in special_handled:
                continue

            if "qkv" in base_key:
                A, B, alpha = (lw.get("A"), lw.get("B"), lw.get("alpha")) if "A" in lw else _fuse_qkv_lora(lw)
            elif ".proj_out" in base_key and "single_transformer_blocks" in base_key:
                split_map, consumed_keys = _handle_proj_out_split(lora_grouped, base_key, model)
                processed_groups.update(split_map)
                special_handled.update(consumed_keys)
                continue
            else:
                A, B, alpha = lw.get("A"), lw.get("B"), lw.get("alpha")

            if A is not None and B is not None:
                processed_groups[base_key] = (A, B, alpha)

        for module_key, (A, B, alpha) in processed_groups.items():
            aggregated_weights[module_key].append(
                {"A": A, "B": B, "alpha": alpha, "strength": strength, "source": lora_name})

    # 2. Apply aggregated weights to the model
    applied_modules_count = 0

    for module_name, parts in aggregated_weights.items():
        resolved_name, module = _resolve_module_name(model, module_name)
        if module is None or not (hasattr(module, "proj_down") and hasattr(module, "proj_up")):
            continue

        all_A = []
        all_B_scaled = []
        for part in parts:
            A, B, alpha, strength = part["A"], part["B"], part["alpha"], part["strength"]
            r_lora = A.shape[0]
            scale_alpha = alpha.item() if alpha is not None else float(r_lora)
            scale = strength * (scale_alpha / max(1.0, float(r_lora)))

            if ".norm1.linear" in resolved_name or ".norm1_context.linear" in resolved_name:
                B = reorder_adanorm_lora_up(B, splits=6)
            elif ".single_transformer_blocks." in resolved_name and ".norm.linear" in resolved_name:
                B = reorder_adanorm_lora_up(B, splits=3)

            all_A.append(A.to(dtype=module.proj_down.dtype, device=module.proj_down.device))
            all_B_scaled.append((B * scale).to(dtype=module.proj_up.dtype, device=module.proj_up.device))

        if not all_A:
            continue

        final_A = torch.cat(all_A, dim=0)
        final_B = torch.cat(all_B_scaled, dim=1)

        _apply_lora_to_module(module, final_A, final_B, resolved_name, model)
        applied_modules_count += 1

    logger.info(f"Applied LoRA compositions to {applied_modules_count} modules.")

def update_lora_params_v2(
        model: torch.nn.Module,
        lora_state_dict_or_path: Union[str, Path, Dict[str, torch.Tensor]],
        strength: float = 1.0,
) -> None:
    """Loads and applies a single LoRA to the model (convenience wrapper)."""
    logger.info(f"Loading single LoRA with strength {strength}.")
    compose_loras_v2(model, [(lora_state_dict_or_path, strength)])


def set_lora_strength_v2(model: nn.Module, strength: float) -> None:
    """Adjusts the overall strength of all applied LoRAs as a global multiplier."""
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        logger.warning("No LoRA weights loaded, cannot set strength.")
        return

    old_strength = getattr(model, "_lora_strength", 1.0)
    scale_factor = strength / old_strength if old_strength != 0 else 0

    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None or info.get("appended", 0) <= 0:
            continue

        base_rank, appended = info["base_rank"], info["appended"]
        with torch.no_grad():
            module.proj_up.data[:, base_rank: base_rank + appended] *= scale_factor

    model._lora_strength = strength
    logger.info(f"LoRA global strength updated to {strength}.")


def reset_lora_v2(model: nn.Module) -> None:
    """Removes all appended LoRA weights from the model."""
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        return

    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None:
            continue

        base_rank = info["base_rank"]
        with torch.no_grad():
            pd = unpack_lowrank_weight(module.proj_down.data, down=True)
            pu = unpack_lowrank_weight(module.proj_up.data, down=False)

            if info.get("axis_down", 0) == 0:  # [rank, in]
                pd_reset = pd[:base_rank, :].clone()
            else:  # [in, rank]
                pd_reset = pd[:, :base_rank].clone()
            pu_reset = pu[:, :base_rank].clone()

            module.proj_down.data = pack_lowrank_weight(pd_reset, down=True)
            module.proj_up.data = pack_lowrank_weight(pu_reset, down=False)
            module.rank = base_rank

    model._lora_slots.clear()
    model._lora_strength = 1.0
    logger.info("All LoRA weights have been reset from the model.")