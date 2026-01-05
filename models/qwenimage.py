"""
This module implements the Nunchaku Qwen-Image model and related components.

.. note::

    Inherits and modifies from https://github.com/comfyanonymous/ComfyUI/blob/v0.3.51/comfy/ldm/qwen_image/model.py

.. warning::

    There is a sage-attention dispatch bug that may cause black images until the upstream issue is fixed.
    See: https://github.com/comfyanonymous/ComfyUI/issues/9773
"""

import gc
from typing import Optional, Tuple

import torch
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.qwen_image.model import (
    GELU,
    FeedForward,
    LastLayer,
    QwenImageTransformer2DModel,
    QwenTimestepProjEmbeddings,
    apply_rotary_emb,
)
from torch import nn

from nunchaku.models.linear import AWQW4A16Linear, SVDQW4A4Linear
from nunchaku.models.utils import CPUOffloadManager
from nunchaku.ops.fused import fused_gelu_mlp

from ..mixins.model import NunchakuModelMixin


class NunchakuGELU(GELU):
    """
    GELU activation with a quantized linear projection.

    Parameters
    ----------
    dim_in : int
        Input feature dimension.
    dim_out : int
        Output feature dimension.
    approximate : str, optional
        Approximation mode for GELU (default: "none").
    bias : bool, optional
        Whether to use bias in the projection (default: True).
    dtype : torch.dtype, optional
        Data type for the projection.
    device : torch.device, optional
        Device for the projection.
    **kwargs
        Additional arguments for the quantized linear layer.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
        dtype=None,
        device=None,
        **kwargs,
    ):
        super(GELU, self).__init__()
        self.proj = SVDQW4A4Linear(dim_in, dim_out, bias=bias, torch_dtype=dtype, device=device, **kwargs)
        self.approximate = approximate


class NunchakuFeedForward(FeedForward):
    """
    Feed-forward network with fused quantized layers and optional fused GELU-MLP.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    dim_out : int, optional
        Output feature dimension. If None, set to `dim`.
    mult : int, optional
        Expansion factor for the hidden layer (default: 4).
    dropout : float, optional
        Dropout probability (default: 0.0).
    inner_dim : int, optional
        Hidden layer dimension. If None, computed as `dim * mult`.
    bias : bool, optional
        Whether to use bias in the projections (default: True).
    dtype : torch.dtype, optional
        Data type for the projections.
    device : torch.device, optional
        Device for the projections.
    **kwargs
        Additional arguments for the quantized linear layers.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        inner_dim=None,
        bias: bool = True,
        dtype=None,
        device=None,
        **kwargs,
    ):
        super(FeedForward, self).__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        self.net.append(
            NunchakuGELU(dim, inner_dim, approximate="tanh", bias=bias, dtype=dtype, device=device, **kwargs)
        )
        self.net.append(nn.Dropout(dropout))
        self.net.append(
            SVDQW4A4Linear(
                inner_dim,
                dim_out,
                bias=bias,
                act_unsigned=kwargs["precision"]
                == "int4",  # For int4 quantization, the second linear layer is unsigned as the output of the first is shifted positive in fused_gelu_mlp
                torch_dtype=dtype,
                device=device,
                **kwargs,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feed-forward network.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape (batch, seq_len, dim).

        Returns
        -------
        torch.Tensor
            Output tensor after feed-forward transformation.
        """
        if isinstance(self.net[0], NunchakuGELU):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        else:
            # Fallback to original implementation
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states


class Attention(nn.Module):
    """
    Double-stream attention module for joint image-text attention.

    This module fuses QKV projections for both image and text streams for improved speed,
    applies Q/K normalization and rotary embeddings, and computes joint attention.

    Parameters
    ----------
    query_dim : int
        Input feature dimension.
    dim_head : int, optional
        Dimension per attention head (default: 64).
    heads : int, optional
        Number of attention heads (default: 8).
    dropout : float, optional
        Dropout probability (default: 0.0).
    bias : bool, optional
        Whether to use bias in projections (default: False).
    eps : float, optional
        Epsilon for normalization layers (default: 1e-5).
    out_bias : bool, optional
        Whether to use bias in output projections (default: True).
    out_dim : int, optional
        Output dimension for image stream.
    out_context_dim : int, optional
        Output dimension for text stream.
    dtype : torch.dtype, optional
        Data type for projections.
    device : torch.device, optional
        Device for projections.
    operations : module, optional
        Module providing normalization and linear layers.
    **kwargs
        Additional arguments for quantized linear layers.
    """

    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_bias: bool = True,
        out_dim: int = None,
        out_context_dim: int = None,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.heads = heads
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.dropout = dropout

        # Q/K normalization for both streams
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        # Image stream projections: fused QKV for speed
        self.to_qkv = SVDQW4A4Linear(
            query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, torch_dtype=dtype, device=device, **kwargs
        )

        # Text stream projections: fused QKV for speed
        self.add_qkv_proj = SVDQW4A4Linear(
            query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, torch_dtype=dtype, device=device, **kwargs
        )

        # Output projections
        self.to_out = nn.ModuleList(
            [
                SVDQW4A4Linear(self.inner_dim, self.out_dim, bias=out_bias, torch_dtype=dtype, device=device, **kwargs),
                nn.Dropout(dropout),
            ]
        )
        self.to_add_out = SVDQW4A4Linear(
            self.inner_dim, self.out_context_dim, bias=out_bias, torch_dtype=dtype, device=device, **kwargs
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for double-stream attention.

        Parameters
        ----------
        hidden_states : torch.FloatTensor
            Image stream input tensor of shape (batch, seq_len_img, dim).
        encoder_hidden_states : torch.FloatTensor, optional
            Text stream input tensor of shape (batch, seq_len_txt, dim).
        encoder_hidden_states_mask : torch.FloatTensor, optional
            Mask for encoder hidden states.
        attention_mask : torch.FloatTensor, optional
            Attention mask for joint attention.
        image_rotary_emb : torch.Tensor, optional
            Rotary positional embeddings.

        Returns
        -------
        img_attn_output : torch.Tensor
            Output tensor for image stream.
        txt_attn_output : torch.Tensor
            Output tensor for text stream.
        """
        seq_txt = encoder_hidden_states.shape[1]

        img_qkv = self.to_qkv(hidden_states)
        img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)

        # Compute QKV for text stream (context projections)
        txt_qkv = self.add_qkv_proj(encoder_hidden_states)
        txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)

        img_query = img_query.unflatten(-1, (self.heads, -1))
        img_key = img_key.unflatten(-1, (self.heads, -1))
        img_value = img_value.unflatten(-1, (self.heads, -1))

        txt_query = txt_query.unflatten(-1, (self.heads, -1))
        txt_key = txt_key.unflatten(-1, (self.heads, -1))
        txt_value = txt_value.unflatten(-1, (self.heads, -1))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        # Concatenate image and text streams for joint attention
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Apply rotary embeddings
        joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
        joint_key = apply_rotary_emb(joint_key, image_rotary_emb)

        joint_query = joint_query.flatten(start_dim=2)
        joint_key = joint_key.flatten(start_dim=2)
        joint_value = joint_value.flatten(start_dim=2)

        # Compute joint attention
        joint_hidden_states = optimized_attention_masked(
            joint_query, joint_key, joint_value, self.heads, attention_mask
        )

        # Split results back to separate streams
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class NunchakuQwenImageTransformerBlock(nn.Module):
    """
    Transformer block with dual-stream (image/text) processing, modulation, and quantized attention/MLP.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    num_attention_heads : int
        Number of attention heads.
    attention_head_dim : int
        Dimension per attention head.
    eps : float, optional
        Epsilon for normalization layers (default: 1e-6).
    dtype : torch.dtype, optional
        Data type for projections.
    device : torch.device, optional
        Device for projections.
    operations : module, optional
        Module providing normalization and linear layers.
    scale_shift : float, optional
        Value added to scale in modulation (default: 1.0). Nunchaku may have fused the scale's shift into bias.
    **kwargs
        Additional arguments for quantized linear layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None,
        scale_shift: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.scale_shift = scale_shift
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Modulation and normalization for image stream
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            AWQW4A16Linear(dim, 6 * dim, bias=True, torch_dtype=dtype, device=device),
        )
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, **kwargs)

        # Modulation and normalization for text stream
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            AWQW4A16Linear(dim, 6 * dim, bias=True, torch_dtype=dtype, device=device),
        )
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, **kwargs)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            dtype=dtype,
            device=device,
            operations=operations,
            **kwargs,
        )

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply modulation to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, dim).
        mod_params : torch.Tensor
            Modulation parameters of shape (batch, 3*dim).

        Returns
        -------
        modulated_x : torch.Tensor
            Modulated tensor.
        gate : torch.Tensor
            Gate tensor for residual connection.
        """
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if self.scale_shift != 0:
            scale.add_(self.scale_shift)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep_zero_index=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the transformer block.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Image stream input tensor.
        encoder_hidden_states : torch.Tensor
            Text stream input tensor.
        encoder_hidden_states_mask : torch.Tensor
            Mask for encoder hidden states.
        temb : torch.Tensor
            Timestep or conditioning embedding.
        image_rotary_emb : tuple of torch.Tensor, optional
            Rotary positional embeddings.

        Returns
        -------
        encoder_hidden_states : torch.Tensor
            Updated text stream tensor.
        hidden_states : torch.Tensor
            Updated image stream tensor.
        """
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Nunchaku's mod_params is [B, 6*dim] instead of [B, dim*6]
        img_mod_params = (
            img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
        )
        txt_mod_params = (
            txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)
        )

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Joint attention computation (DoubleStreamLayerMegatron logic)
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream ("sample")
            encoder_hidden_states=txt_modulated,  # Text stream ("context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, hidden_states


class NunchakuQwenImageTransformer2DModel(NunchakuModelMixin, QwenImageTransformer2DModel):
    """
    Full transformer model for QwenImage, using Nunchaku-optimized blocks.

    Parameters
    ----------
    patch_size : int, optional
        Patch size for image input (default: 2).
    in_channels : int, optional
        Number of input channels (default: 64).
    out_channels : int, optional
        Number of output channels (default: 16).
    num_layers : int, optional
        Number of transformer layers (default: 60).
    attention_head_dim : int, optional
        Dimension per attention head (default: 128).
    num_attention_heads : int, optional
        Number of attention heads (default: 24).
    joint_attention_dim : int, optional
        Dimension for joint attention (default: 3584).
    pooled_projection_dim : int, optional
        Dimension for pooled projection (default: 768).
    guidance_embeds : bool, optional
        Whether to use guidance embeddings (default: False).
    axes_dims_rope : tuple of int, optional
        Axes dimensions for rotary embeddings (default: (16, 56, 56)).
    image_model : module, optional
        Optional image model.
    dtype : torch.dtype, optional
        Data type for projections.
    device : torch.device, optional
        Device for projections.
    operations : module, optional
        Module providing normalization and linear layers.
    scale_shift : float, optional
        Value added to scale in modulation (default: 1.0).
    transformer_offload_device: torch.device, optional
        If not None, transformer blocks will be initialized to this device (usually cpu) rather than `device`
    **kwargs
        Additional arguments for quantized linear layers.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 768,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        default_ref_method="index",
        image_model=None,
        final_layer=True,
        use_additional_t_cond=False,
        dtype=None,
        device=None,
        operations=None,
        scale_shift: float = 1.0,
        transformer_offload_device=None,
        **kwargs,
    ):
        super(QwenImageTransformer2DModel, self).__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.default_ref_method = default_ref_method

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            use_additional_t_cond=use_additional_t_cond,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.txt_norm = operations.RMSNorm(joint_attention_dim, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(in_channels, self.inner_dim, dtype=dtype, device=device)
        self.txt_in = operations.Linear(joint_attention_dim, self.inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [
                NunchakuQwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    device=transformer_offload_device if transformer_offload_device is not None else device,
                    operations=operations,
                    scale_shift=scale_shift,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        if final_layer:
            self.norm_out = LastLayer(
                self.inner_dim,
                self.inner_dim,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            self.proj_out = operations.Linear(
                self.inner_dim,
                patch_size * patch_size * self.out_channels,
                bias=True,
                dtype=dtype,
                device=device,
            )
        self.gradient_checkpointing = False

    def _forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        ref_latents=None,
        additional_t_cond=None,
        transformer_options={},
        control=None,
        **kwargs,
    ):
        """
        Forward pass of the Nunchaku Qwen-Image model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, channels, height, width).
        timesteps : torch.Tensor or int
            Timestep(s) for diffusion process.
        context : torch.Tensor
            Textual context tensor (e.g., from a text encoder).
        attention_mask : torch.Tensor, optional
            Optional attention mask for the context.
        guidance : torch.Tensor, optional
            Optional guidance tensor for classifier-free guidance.
        ref_latents : list[torch.Tensor], optional
            Optional list of reference latent tensors for multi-image conditioning.
        transformer_options : dict, optional
            Dictionary of options for transformer block patching and replacement.
        **kwargs
            Additional keyword arguments. Supports 'ref_latents_method' to control reference latent handling.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, channels, height, width), matching the input spatial dimensions.

        """
        device = x.device
        if self.offload:
            self.offload_manager.set_device(device)

        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        timestep_zero_index = None
        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            ref_method = kwargs.get("ref_latents_method", self.default_ref_method)
            index_ref_method = (ref_method == "index") or (ref_method == "index_timestep_zero")
            negative_ref_method = ref_method == "negative_index"
            timestep_zero = ref_method == "index_timestep_zero"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
                elif negative_ref_method:
                    index -= 1
                    h_offset = 0
                    w_offset = 0
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
            if timestep_zero:
                if index > 0:
                    timestep = torch.cat([timestep, timestep * 0], dim=0)
                    timestep_zero_index = num_embeds

        txt_start = round(
            max(
                ((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2,
                ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2,
            )
        )
        txt_ids = (
            torch.arange(txt_start, txt_start + context.shape[1], device=x.device)
            .reshape(1, -1, 1)
            .repeat(x.shape[0], 1, 3)
        )
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
        del ids, txt_ids, img_ids

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        # Setup compute stream for offloading
        compute_stream = torch.cuda.current_stream()
        if self.offload:
            self.offload_manager.initialize(compute_stream)

        for i, block in enumerate(self.transformer_blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload:
                    block = self.offload_manager.get_block(i)
                if ("double_block", i) in blocks_replace:

                    def block_wrap(args):
                        out = {}
                        out["txt"], out["img"] = block(
                            hidden_states=args["img"],
                            encoder_hidden_states=args["txt"],
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=args["vec"],
                            image_rotary_emb=args["pe"],
                        )
                        return out

                    out = blocks_replace[("double_block", i)](
                        {"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb},
                        {"original_block": block_wrap},
                    )
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        timestep_zero_index=timestep_zero_index,
                    )
                # ControlNet helpers(device/dtype-safe residual adds)
                _control = (
                    control
                    if control is not None
                    else (transformer_options.get("control", None) if isinstance(transformer_options, dict) else None)
                )
                if isinstance(_control, dict):
                    control_i = _control.get("input")
                    try:
                        _scale = float(_control.get("weight", _control.get("scale", 1.0)))
                    except Exception:
                        _scale = 1.0
                else:
                    control_i = None
                    _scale = 1.0
                if control_i is not None and i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        if (
                            getattr(add, "device", None) != hidden_states.device
                            or getattr(add, "dtype", None) != hidden_states.dtype
                        ):
                            add = add.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
                        t = min(hidden_states.shape[1], add.shape[1])
                        if t > 0:
                            hidden_states[:, :t].add_(add[:, :t], alpha=_scale)

            if self.offload:
                self.offload_manager.step(compute_stream)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(
            orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2
        )
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, : x.shape[-2], : x.shape[-1]]

    def set_offload(self, offload: bool, **kwargs):
        """
        Enable or disable CPU offloading for the transformer blocks.

        Parameters
        ----------
        offload : bool
            If True, enable CPU offloading. If False, disable it.
        **kwargs
            Additional keyword arguments:
                - use_pin_memory (bool): Whether to use pinned memory (default: True).
                - num_blocks_on_gpu (int): Number of transformer blocks to keep on GPU (default: 1).

        Notes
        -----
        - When offloading is enabled, only a subset of modules remain on GPU.
        - When disabling, memory is released and CUDA cache is cleared.
        """
        if offload == self.offload:
            # Nothing changed, just return
            return
        self.offload = offload
        if offload:
            self.offload_manager = CPUOffloadManager(
                self.transformer_blocks,
                use_pin_memory=kwargs.get("use_pin_memory", True),
                on_gpu_modules=[
                    self.img_in,
                    self.txt_in,
                    self.txt_norm,
                    self.time_text_embed,
                    self.norm_out,
                    self.proj_out,
                ],
                num_blocks_on_gpu=kwargs.get("num_blocks_on_gpu", 1),
            )
        else:
            self.offload_manager = None
            gc.collect()
            torch.cuda.empty_cache()
