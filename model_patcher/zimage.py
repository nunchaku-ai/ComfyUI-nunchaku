import logging
import uuid
from collections import namedtuple

import comfy.model_management
import comfy.utils
import torch
import torch.nn as nn
from comfy.float import stochastic_rounding
from comfy.ldm.lumina.model import NextDiT
from comfy.lora import calculate_weight, model_lora_keys_unet
from comfy.model_base import BaseModel
from comfy.model_management import cast_to_device, lora_compute_dtype
from comfy.model_patcher import LowVramPatch, ModelPatcher, get_key_weight, move_weight_functions, string_to_seed
from comfy.utils import get_attr, set_attr_param
from comfy.weight_adapter.lora import LoRAAdapter

from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight, unpack_lowrank_weight
from nunchaku.models.linear import SVDQW4A4Linear


def apply_lora_to_svdq_linear(linear: SVDQW4A4Linear, lora_down: torch.Tensor, lora_up: torch.Tensor):
    """
    Apply LoRA weights into quantized SVDQW4A4Linear module.

    Parameters
    ----------
    linear : SVDQW4A4Linear
        The Nunchaku quantized module to apply LoRAs to
    lora_down : torch.Tensor
        The down projection part of the LoRA weights.
    lora_up : torch.Tensor
        The up projection part of the LoRA weights.
    """
    dtype = linear.proj_down.dtype
    proj_down = unpack_lowrank_weight(linear.proj_down, down=True)
    proj_up = unpack_lowrank_weight(linear.proj_up, down=False)
    concatenated_down = torch.cat([proj_down, lora_down], dim=0).to(dtype=dtype)
    concatenated_up = torch.cat([proj_up, lora_up], dim=1).to(dtype=dtype)

    linear.proj_down = nn.Parameter(pack_lowrank_weight(concatenated_down, down=True))
    linear.proj_up = nn.Parameter(pack_lowrank_weight(concatenated_up, down=False))


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


class ZImageModelPatcher(ModelPatcher):
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        """
        Adapted from comfy.model_patcher.ModelPatcher#clone

        Note
        ----
        + Always set `weight_inplace_update` to False
        + Add `svdq_backup` dict for loading/unloading lora weights for Nunchaku Z-Image model.
        """
        super().__init__(model, load_device, offload_device, size, weight_inplace_update=False)
        self.svdq_backup = {}

    def clone(self):
        """
        Adapted from comfy.model_patcher.ModelPatcher#clone

        Note
        ----
        + Clone quantized svdq_backup weights for Nunchaku Z-Image model.
        """
        n = ModelPatcher.clone(self)
        n.svdq_backup = self.svdq_backup
        logging.debug("ZImageModelPatcher cloned.")
        return n

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """
        Adapted from comfy.model_patcher.ModelPatcher#unpatch_model

        Note
        ----
        + Unpatch loras and restore original weights from svdq_backup for Nunchaku Z-Image model.
        """
        super().unpatch_model(device_to, unpatch_weights)
        if unpatch_weights:
            for k, (proj_down, proj_up) in self.svdq_backup.items():
                logging.debug(f"<<<< Unpatching svdq_linears, key: {k}, device_to: {device_to}")
                attr_name = k.replace(".qweight", "")
                linear = get_attr(self.model, attr_name)
                assert isinstance(linear, SVDQW4A4Linear)
                if device_to is not None:
                    proj_down, proj_up = proj_down.to(device=device_to), proj_up.to(device=device_to)
                linear.proj_down = proj_down
                linear.proj_up = proj_up
            self.svdq_backup.clear()

    def do_svdq_linear_backup(self, key: str, linear: SVDQW4A4Linear):
        assert key.endswith(".qweight"), f">>> Unexpected key {key} to do svdq_linear backup"
        if key not in self.svdq_backup:
            self.svdq_backup[key] = (linear.proj_down, linear.proj_up)

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        """
        Adapted from comfy.model_patcher.ModelPatcher#partially_unload

        Note
        ----
        + Unload quantized svdq linears for Nunchaku Z-Image model.
        """
        with self.use_ejected():
            hooks_unpatched = False
            memory_freed = 0
            patch_counter = 0
            unload_list = self._load_list()
            unload_list.sort()

            offload_buffer = self.model.model_offload_buffer_memory
            if len(unload_list) > 0:
                NS = comfy.model_management.NUM_STREAMS
                offload_weight_factor = [min(offload_buffer / (NS + 1), unload_list[0][1])] * NS

            for unload in unload_list:
                if memory_to_free + offload_buffer - self.model.model_offload_buffer_memory < memory_freed:
                    break
                module_offload_mem, module_mem, n, m, params = unload

                potential_offload = module_offload_mem + sum(offload_weight_factor)

                lowvram_possible = hasattr(m, "comfy_cast_weights")
                if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights:
                    move_weight = True
                    for param in params:
                        key = "{}.{}".format(n, param)
                        bk = self.backup.get(key, None)
                        svdq_bk = self.svdq_backup.get(key, None)
                        if bk is not None or svdq_bk is not None:
                            if not lowvram_possible:
                                move_weight = False
                                break

                            if not hooks_unpatched:
                                self.unpatch_hooks()
                                hooks_unpatched = True
                            if bk is not None:
                                if bk.inplace_update:
                                    comfy.utils.copy_to_param(self.model, key, bk.weight)
                                else:
                                    comfy.utils.set_attr_param(self.model, key, bk.weight)
                                self.backup.pop(key)
                            # region unload svdq linears
                            if svdq_bk is not None:
                                logging.debug(f">>> partially unload svdq, key: {key}")
                                proj_down, proj_up = svdq_bk
                                attr_name = key.replace(".qweight", "")
                                linear = get_attr(self.model, attr_name)
                                assert isinstance(linear, SVDQW4A4Linear)
                                linear.proj_down = proj_down
                                linear.proj_up = proj_up
                                self.svdq_backup.pop(key)
                            # end of region

                    weight_key = "{}.weight".format(n)
                    bias_key = "{}.bias".format(n)
                    if move_weight:
                        cast_weight = self.force_cast_weights
                        m.to(device_to)
                        module_mem += move_weight_functions(m, device_to)
                        if lowvram_possible:
                            if weight_key in self.patches:
                                if force_patch_weights:
                                    self.patch_weight_to_device(weight_key)
                                else:
                                    _, set_func, convert_func = get_key_weight(self.model, weight_key)
                                    m.weight_function.append(
                                        LowVramPatch(weight_key, self.patches, convert_func, set_func)
                                    )
                                    patch_counter += 1
                            if bias_key in self.patches:
                                if force_patch_weights:
                                    self.patch_weight_to_device(bias_key)
                                else:
                                    _, set_func, convert_func = get_key_weight(self.model, bias_key)
                                    m.bias_function.append(LowVramPatch(bias_key, self.patches, convert_func, set_func))
                                    patch_counter += 1
                            cast_weight = True

                        if cast_weight and hasattr(m, "comfy_cast_weights"):
                            m.prev_comfy_cast_weights = m.comfy_cast_weights
                            m.comfy_cast_weights = True
                        m.comfy_patched_weights = False
                        memory_freed += module_mem
                        offload_buffer = max(offload_buffer, potential_offload)
                        offload_weight_factor.append(module_mem)
                        offload_weight_factor.pop(0)
                        logging.debug("freed {}".format(n))

                        for param in params:
                            self.pin_weight_to_device("{}.{}".format(n, param))

            self.model.model_lowvram = True
            self.model.lowvram_patch_counter += patch_counter
            self.model.model_loaded_weight_memory -= memory_freed
            self.model.model_offload_buffer_memory = offload_buffer
            logging.info(
                "Unloaded partially: {:.2f} MB freed, {:.2f} MB remains loaded, {:.2f} MB buffer reserved, lowvram patches: {}".format(
                    memory_freed / (1024 * 1024),
                    self.model.model_loaded_weight_memory / (1024 * 1024),
                    offload_buffer / (1024 * 1024),
                    self.model.lowvram_patch_counter,
                )
            )
            return memory_freed

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        """
        Adapted from comfy.model_patcher.ModelPatcher#add_patches

        Note
        ----
        + Modify the valid key_set
        """
        with self.use_ejected():
            p = set()
            key_set = set(model_lora_keys_unet(self.model).values())
            for k in patches:
                offset = None
                function = None
                if isinstance(k, str):
                    key = k
                else:
                    offset = k[1]
                    key = k[0]
                    if len(k) > 2:
                        function = k[2]
                if k in key_set:
                    p.add(k)
                    current_patches = self.patches.get(key, [])
                    current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                    self.patches[key] = current_patches

            self.patches_uuid = uuid.uuid4()
            return list(p)

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        """
        Adapted from comfy.model_patcher.ModelPatcher#patch_weight_to_device

        Note
        ----
        + Patch non-quantized linear modules the same way as original ModelPatcher
        + Patch quantized svdq_linear modules by concatenate loras into proj_down and proj_up.
        """
        assert isinstance(self.model, BaseModel)
        assert isinstance(self.model.diffusion_model, NextDiT)
        assert isinstance(key, str)
        assert (
            inplace_update is False
        ), "In-place weight update is not supported for applying LoRA patches to Nunchaku Z-Image models."

        if len(self.patches) == 0:
            return

        # the `key` argument passed in could be a name of any nn.Parameter in the model
        if (key not in self.patches) and not (key.endswith(".qweight")):
            return

        assert key.endswith(".qweight") or key.endswith(".weight"), f"Unexpected key: {key}"
        attr_name = key.rsplit(".", 1)[0]
        linear = get_attr(self.model, attr_name)

        if isinstance(linear, nn.Linear):
            logging.debug(
                f">>> Applying LoRA to non-quantized linear at {attr_name}, linear weight shape: {linear.weight.shape}, number of patches: {len(self.patches[key])}"
            )
            # Apply LoRA to standard linear layer. This is the same way as in comfy.model_patcher.ModelPatcher#patch_weight_to_device
            weight = linear.weight
            if key not in self.backup:
                self.backup[key] = namedtuple("Dimension", ["weight", "inplace_update"])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )
            temp_dtype = lora_compute_dtype(device_to)
            if device_to is not None:
                temp_weight = cast_to_device(weight, device_to, temp_dtype, copy=True)
            else:
                temp_weight = weight.to(temp_dtype, copy=True)
            out_weight = calculate_weight(self.patches[key], temp_weight, key)
            out_weight = stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
            set_attr_param(self.model, key, out_weight)
        elif isinstance(linear, SVDQW4A4Linear):

            def _parse_patch(_the_patch):
                strength = _the_patch[0]
                lora_adapter: LoRAAdapter = _the_patch[1]
                up, down, alpha, _, _, _ = lora_adapter.weights
                rank = down.shape[0]
                return strength, up, down, alpha, rank

            if ".qkv." in key:
                patches_list_for_key = self.patches.get(key.replace(".qweight", ".weight"), [])
                logging.debug(
                    f">>> Applying LoRA to fused qkv at {key}, svdq_linear shape: {(linear.in_features, linear.out_features)}, svdq_linear rank: {linear.rank}, number of patches: {len(patches_list_for_key)}"
                )
                if len(patches_list_for_key) > 0:
                    q_patches, k_patches, v_patches = [], [], []
                    for strength_patch, lora_adapter, strength_model, offset, function in patches_list_for_key:
                        assert isinstance(lora_adapter, LoRAAdapter)
                        assert (
                            offset is not None
                        ), f"Offset must be provided for qkv LoRA patch, but got None for key {key}"
                        if offset[1] == 0:
                            q_patches.append((strength_patch, lora_adapter, strength_model, offset, function))
                        elif offset[1] == offset[2]:
                            k_patches.append((strength_patch, lora_adapter, strength_model, offset, function))
                        elif offset[1] == offset[2] * 2:
                            v_patches.append((strength_patch, lora_adapter, strength_model, offset, function))
                        else:
                            raise ValueError(f"Invalid offset {offset} for qkv LoRA patch in key {key}")

                    assert len(q_patches) == len(k_patches) == len(v_patches)
                    composed_up, composed_down = None, None
                    for q_patch, k_patch, v_patch in zip(q_patches, k_patches, v_patches):
                        q_strength, q_up, q_down, q_alpha, q_rank = _parse_patch(q_patch)
                        k_strength, k_up, k_down, k_alpha, k_rank = _parse_patch(k_patch)
                        v_strength, v_up, v_down, v_alpha, v_rank = _parse_patch(v_patch)
                        composed_down, composed_up = concat_lora_weights(
                            composed_down,
                            composed_up,
                            [q_down, k_down, v_down],
                            [q_up, k_up, v_up],
                            [
                                q_strength * (q_alpha / q_rank) if q_alpha is not None else q_strength,
                                k_strength * (k_alpha / k_rank) if k_alpha is not None else k_strength,
                                v_strength * (v_alpha / v_rank) if v_alpha is not None else v_strength,
                            ],
                        )
                    self.do_svdq_linear_backup(key, linear)
                    apply_lora_to_svdq_linear(linear, composed_down, composed_up)
            elif ".feed_forward.w13." in key:
                w1_patches = self.patches.get(key.replace("w13.qweight", "w1.weight"), [])
                w3_patches = self.patches.get(key.replace("w13.qweight", "w3.weight"), [])
                assert len(w1_patches) == len(w3_patches)
                logging.debug(
                    f">>> Applying LoRA to fused ff.w13 at {key}, svdq_linear shape: {(linear.in_features, linear.out_features)}, svdq_linear rank: {linear.rank}, number of patches: {len(w1_patches)}"
                )
                if len(w1_patches) > 0:
                    composed_up, composed_down = None, None
                    for w1_patch, w3_patch in zip(w1_patches, w3_patches):
                        w1_strength, w1_up, w1_down, w1_alpha, w1_rank = _parse_patch(w1_patch)
                        w3_strength, w3_up, w3_down, w3_alpha, w3_rank = _parse_patch(w3_patch)
                        composed_down, composed_up = concat_lora_weights(
                            composed_down,
                            composed_up,
                            [w3_down, w1_down],
                            [w3_up, w1_up],
                            [
                                w3_strength * (w3_alpha / w3_rank) if w3_alpha is not None else w3_strength,
                                w1_strength * (w1_alpha / w1_rank) if w1_alpha is not None else w1_strength,
                            ],
                        )
                    self.do_svdq_linear_backup(key, linear)
                    apply_lora_to_svdq_linear(linear, composed_down, composed_up)
            else:
                patches_list_for_key = self.patches.get(key.replace(".qweight", ".weight"), [])
                logging.debug(
                    f">>> Applying LoRA to non-fused quantized key {key}, svdq_linear shape: {(linear.in_features, linear.out_features)}, svdq_linear rank: {linear.rank}, number of patches: {len(patches_list_for_key)}"
                )
                if len(patches_list_for_key) > 0:
                    composed_up, composed_down = None, None
                    for patch in patches_list_for_key:
                        strength, up, down, alpha, rank = _parse_patch(patch)
                        composed_down, composed_up = concat_lora_weights(
                            composed_down,
                            composed_up,
                            [down],
                            [up],
                            [strength * (alpha / rank) if alpha is not None else strength],
                        )
                    self.do_svdq_linear_backup(key, linear)
                    apply_lora_to_svdq_linear(linear, composed_down, composed_up)
        else:
            raise ValueError(f"Unsupported module type for LoRA patching: {type(linear)}, key: {key}")
