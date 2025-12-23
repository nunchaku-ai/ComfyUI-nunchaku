"""
This module patches the original Z-Image transformer model by injecting into Nunchaku optimized attention and feed forward submodules.

Note
----
The injected transformer model is `comfy.ldm.lumina.model.NextDiT` in ComfyUI repository. Codes are adapted from:
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py#Lumina2
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/lumina/model.py#NextDiT

"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.lumina.model import FeedForward, JointAttention, JointTransformerBlock, NextDiT, clamp_fp16

from nunchaku.models.linear import SVDQW4A4Linear


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
        return super().forward(x, x_mask, freqs_cis, transformer_options)


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

    # `norm_final` is not used in Z-Image-Turbo, prevent state dict loading by setting it to None
    # Maybe remove this line in the future if future Z-Image models use `norm_final`
    diffusion_model.norm_final = None
