"""
This module patches the original Z-Image transformer model by injecting into Nunchaku optimized attention and feed forward submodules.

Note
----
The injected transformer model is `comfy.ldm.lumina.model.NextDiT` in ComfyUI repository. Codes are adapted from:
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py#Lumina2
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/lumina/model.py#NextDiT

"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.lumina.model import FeedForward, JointAttention, JointTransformerBlock, NextDiT, clamp_fp16
from comfy.ldm.modules.attention import optimized_attention_masked

from nunchaku.models.embeddings import pack_rotemb
from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.ops.fused import fused_qkv_norm_rottary
from nunchaku.utils import pad_tensor


def fuse_to_svdquant_linear(comfy_linear1: nn.Linear, comfy_linear2: nn.Linear, **kwargs) -> SVDQW4A4Linear:
    """
    Fuse two linear modules into one SVDQW4A4Linear.

    The two linear modules MUST have equal `in_features`.

    Parameters
    ----------
    comfy_linear1 : nn.Linear
        linear module 1
    comfy_linear2 : nn.Linear
        linear module 2

    Returns
    -------
    SVDQW4A4Linear:
        The fused module.
    """
    assert comfy_linear1.in_features == comfy_linear2.in_features
    assert comfy_linear1.bias is None and comfy_linear2.bias is None
    return SVDQW4A4Linear(
        comfy_linear1.in_features,
        comfy_linear1.out_features + comfy_linear2.out_features,
        bias=False,
        torch_dtype=comfy_linear1.weight.dtype,
        device=comfy_linear1.weight.device,
        **kwargs,
    )


class ComfyNunchakuZImageAttention(JointAttention):
    """
    Nunchaku optimized attention module for ZImage.
    """

    def __init__(self, orig_attn: JointAttention, **kwargs):
        nn.Module.__init__(self)
        self.n_kv_heads = orig_attn.n_kv_heads
        self.n_local_heads = orig_attn.n_local_heads
        self.n_local_kv_heads = orig_attn.n_local_kv_heads
        self.n_rep = orig_attn.n_rep
        self.head_dim = orig_attn.head_dim

        self.qkv = SVDQW4A4Linear.from_linear(orig_attn.qkv, **kwargs)
        self.out = SVDQW4A4Linear.from_linear(orig_attn.out, **kwargs)

        self.q_norm = orig_attn.q_norm
        self.k_norm = orig_attn.k_norm

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        transformer_options={},
    ) -> torch.Tensor:
        """
        Adapted from comfy.ldm.lumina.model.JointAttention#forward

        Optimized with fusing qkv projection, q_norm, k_norm and RoPE in one kernel.
        """
        logging.debug(f"x shape: {x.shape}, freqs_cis shape: {freqs_cis.shape}")
        bsz, seqlen, _ = x.shape
        qkv = fused_qkv_norm_rottary(
            x,
            self.qkv,
            self.q_norm,
            self.k_norm,
            freqs_cis,
        )

        xq, xk, xv = torch.split(
            qkv,
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        output = optimized_attention_masked(
            xq.movedim(1, 2),
            xk.movedim(1, 2),
            xv.movedim(1, 2),
            self.n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        return self.out(output)


class ComfyNunchakuZImageFeedForward(nn.Module):
    """
    Nunchaku optimized feed forward module for ZImage.
    """

    def __init__(self, orig_ff: FeedForward, **kwargs):
        super().__init__()
        self.w13 = fuse_to_svdquant_linear(orig_ff.w1, orig_ff.w3, **kwargs)
        self.w2 = SVDQW4A4Linear.from_linear(orig_ff.w2, **kwargs)

    def _forward_silu_gating(self, x1, x3):
        return clamp_fp16(F.silu(x1) * x3)

    def forward(self, x: torch.Tensor):
        x = self.w13(x)
        x3, x1 = x.chunk(2, dim=-1)
        return self.w2(self._forward_silu_gating(x1, x3))


class RopeFuseAttentionHook:
    """
    This class acts as a pre-forward hook on `ComfyNunchakuZImageAttention`,
    replace the `freqs_cis` tensor with packed one to align with the input format of `fused_qkv_norm_rottary` method.
    """

    def __init__(self):
        self.packed_freqs_cis_cache = {}
        self.hook_handles = []

    def pre_forward(self, module: ComfyNunchakuZImageAttention, input_args: tuple, input_kwargs: dict):
        """
        Pre-forward hook method.
        Create a `packed_freqs_cis` tensor that is aligned with the input format of `fused_qkv_norm_rottary` method
        from the original `freqs_cis` and cache it in the  `packed_freqs_cis_cache` dict.
        """
        new_input_args = list(input_args)
        freqs_cis: torch.Tensor = new_input_args[2]
        if freqs_cis is None:
            return None
        cache_key = (freqs_cis.data_ptr(), freqs_cis.shape)
        orig_shape = freqs_cis.shape
        packed_freqs_cis = self.packed_freqs_cis_cache.get(cache_key, None)
        if packed_freqs_cis is None:
            # freqs_cis shape example: torch.Size([1, 4160, 1, 64, 2, 2])
            # freqs_cis dtype: torch.float32
            freqs_cis = freqs_cis[..., [1], :].squeeze(2)  # See comfy.ldm.flux.math#rope, #apply_rope
            packed_freqs_cis = pack_rotemb(pad_tensor(freqs_cis, 256, 1))
            self.packed_freqs_cis_cache[cache_key] = packed_freqs_cis
            logging.debug(
                f"cache miss and created, cache_key: {cache_key}, orig shape: {orig_shape}, packed shape: {packed_freqs_cis.shape}"
            )
        else:
            logging.debug(
                f"cache hit, cache_key: {cache_key}, orig shape: {orig_shape}, packed shape: {packed_freqs_cis.shape}"
            )
        new_input_args[2] = packed_freqs_cis
        return tuple(new_input_args), input_kwargs

    def hook(self, module: ComfyNunchakuZImageAttention):
        assert isinstance(module, ComfyNunchakuZImageAttention)
        self.hook_handles.append(module.register_forward_pre_hook(self.pre_forward, with_kwargs=True))

    def unhook(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()
        self.packed_freqs_cis_cache.clear()


class RopeFuseTransformerHook:
    """
    This class acts as a (both pre- and post- forward) hook on `comfy.ldm.lumina.model.NextDiT`,
    add hooks to its attention submodules for qkv/norm/RoPE fusion before forward pass,
    and remove the hooks added afterwards.
    """

    def __init__(self, skip_refiners: bool):
        self.skip_refiners = skip_refiners

    def pre_forward(self, module: NextDiT, input_args: tuple):
        """
        Pre-forward hook method.
        Add a newly created `RopeFuseAttentionHook` object to all the attention submodules in the `NextDiT` model.
        """
        self.attn_hook = RopeFuseAttentionHook()
        for _, ly in enumerate(module.layers):
            self.attn_hook.hook(ly.attention)
        if not self.skip_refiners:
            for _, nr in enumerate(module.noise_refiner):
                self.attn_hook.hook(nr.attention)
            for _, cr in enumerate(module.context_refiner):
                self.attn_hook.hook(cr.attention)
        return None

    def post_forward(self, module: NextDiT, input_args: tuple, output: tuple):
        """
        Post-forward hook method.
        Remove the hooks added by `pre_forward` method and clear the cached items.
        """
        self.attn_hook.unhook()
        logging.debug("RopeFuseTransformerHook post_forward called")
        return None

    def hook(self, model: NextDiT):
        assert isinstance(model, NextDiT)
        self.pre_handle = model.register_forward_pre_hook(self.pre_forward)
        self.post_handle = model.register_forward_hook(self.post_forward, always_call=True)


def patch_model(diffusion_model: NextDiT, skip_refiners: bool, **kwargs):
    """
    Patch the ZImage diffusion model by replacing the attention and feed forward modules with Nunchaku optimized ones in the transformer blocks.

    Parameters
    ----------
    diffusion_model : NextDiT
        The ZImage diffusion model to be patched.
    skip_refiners : bool
        If true, `noise_refiner` and `context_refiner` will NOT be replaced.
    """

    def _patch_transformer_block(block_list: List[JointTransformerBlock]):
        for _, block in enumerate(block_list):
            block.attention = ComfyNunchakuZImageAttention(block.attention, **kwargs)
            block.feed_forward = ComfyNunchakuZImageFeedForward(block.feed_forward, **kwargs)

    _patch_transformer_block(diffusion_model.layers)
    if not skip_refiners:
        _patch_transformer_block(diffusion_model.noise_refiner)
        _patch_transformer_block(diffusion_model.context_refiner)
    RopeFuseTransformerHook(skip_refiners).hook(diffusion_model)

    # `norm_final` is not used in Z-Image-Turbo, prevent state dict loading by setting it to None
    # Maybe remove this line in the future if future Z-Image models use `norm_final`
    diffusion_model.norm_final = None
