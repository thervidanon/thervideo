"""TherVideo transformer: Wan2.2-TI2V-5B with in-context video conditioning.

The clean RGB condition video and the noisy thermal target video are patchified with the same patch embedding and
concatenated along the token axis:

    [ condition tokens (t = 0) | target tokens (t) ]

Both streams share the same 3D rotary positions, so condition and target tokens are pixel-aligned. A learned
modality embedding is added to every token of each stream. Only the target tokens are decoded.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x2 = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x2[..., 0], x2[..., 1]
    cos, sin = cos[..., 0::2], sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


class AttnProcessor:
    """Scaled dot-product attention with query/key RMS norm and optional rotary embedding."""

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q, k, v = attn.to_q(hidden_states), attn.to_k(context), attn.to_v(context)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        q, k, v = (t.unflatten(2, (attn.heads, -1)).transpose(1, 2) for t in (q, k, v))
        if rotary_emb is not None:
            q, k = _apply_rope(q, *rotary_emb), _apply_rope(k, *rotary_emb)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).flatten(2, 3).type_as(q)
        return attn.to_out[1](attn.to_out[0](out))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, qk_norm: str, cross_attn_norm: bool, eps: float):
        super().__init__()
        attn_kw = dict(query_dim=dim, heads=num_heads, kv_heads=num_heads, dim_head=dim // num_heads, qk_norm=qk_norm,
                       eps=eps, bias=True, cross_attention_dim=None, out_bias=True, processor=AttnProcessor())
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(**attn_kw)
        self.attn2 = Attention(**attn_kw, added_kv_proj_dim=None, added_proj_bias=True)
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, context, temb, rotary_emb):
        # temb: [B, S, 6, D] per-token modulation
        shift, scale, gate, c_shift, c_scale, c_gate = (
            t.squeeze(2) for t in (self.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2))
        h = (self.norm1(x.float()) * (1 + scale) + shift).type_as(x)
        x = (x.float() + self.attn1(hidden_states=h, rotary_emb=rotary_emb) * gate).type_as(x)
        h = self.norm2(x.float()).type_as(x)
        x = x + self.attn2(hidden_states=h, encoder_hidden_states=context)
        h = (self.norm3(x.float()) * (1 + c_scale) + c_shift).type_as(x)
        return (x.float() + self.ffn(h).float() * c_gate).type_as(x)


class ConditionEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int, text_dim: int):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, dim * 6)
        self.text_embedder = PixArtAlphaTextProjection(text_dim, dim, act_fn="gelu_tanh")

    def forward(self, timestep: torch.Tensor, text: torch.Tensor, seq_len: int):
        t = self.timesteps_proj(timestep).unflatten(0, (-1, seq_len))
        t = t.to(next(self.time_embedder.parameters()).dtype)
        temb = self.time_embedder(t).type_as(text)
        return temb, self.time_proj(self.act_fn(temb)), self.text_embedder(text)


class RotaryEmbed(nn.Module):
    """3D rotary embedding over (frame, height, width) token positions."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim, self.theta = head_dim, theta
        self.dtype = torch.float32

    def forward(self, grid: Tuple[int, int, int], device) -> Tuple[torch.Tensor, torch.Tensor]:
        f, h, w = grid
        axes = torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float32) for n in (f, h, w)), indexing="ij")
        pos = torch.stack(axes, -1).reshape(-1, 3)
        dims = (self.head_dim - 2 * (self.head_dim // 3), self.head_dim // 3, self.head_dim // 3)
        cos, sin = [], []
        for axis, dim in enumerate(dims):
            inv = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=device, dtype=self.dtype) / float(dim)))
            fr = pos[:, axis].to(self.dtype)[:, None] * inv[None]
            cos.append(fr.cos().repeat_interleave(2, -1))
            sin.append(fr.sin().repeat_interleave(2, -1))
        n = pos.shape[0]
        return torch.cat(cos, -1).reshape(1, 1, n, -1), torch.cat(sin, -1).reshape(1, 1, n, -1)


class TherVideoTransformer(ModelMixin, ConfigMixin):
    """Loads the Wan2.2-TI2V-5B transformer weights (diffusers layout) unchanged."""

    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["TransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(self, patch_size=(1, 2, 2), num_attention_heads=24, attention_head_dim=128, in_channels=48,
                 out_channels=48, text_dim=4096, freq_dim=256, ffn_dim=14336, num_layers=30, cross_attn_norm=True,
                 qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None, added_kv_proj_dim=None,
                 rope_max_seq_len=1024, pos_embed_seq_len=None):
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.rope = RotaryEmbed(attention_head_dim)
        self.patch_embedding = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.condition_embedder = ConditionEmbedder(dim, freq_dim, text_dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps)
                                     for _ in range(num_layers)])
        self.norm_out = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        # modality embeddings of the RGB condition and the thermal target streams (loaded from the TherVideo checkpoint)
        self.register_buffer("embed_rgb", torch.zeros(dim), persistent=False)
        self.register_buffer("embed_thermal", torch.zeros(dim), persistent=False)

    def _patchify(self, x: torch.Tensor):
        p_t, p_h, p_w = self.config.patch_size
        grid = (x.shape[2] // p_t, x.shape[3] // p_h, x.shape[4] // p_w)
        return self.patch_embedding(x).flatten(2).transpose(1, 2), grid

    def _unpatchify(self, tokens: torch.Tensor, b: int, grid):
        p_t, p_h, p_w = self.config.patch_size
        f, h, w = grid
        x = tokens.reshape(b, f, h, w, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, timestep: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Velocity of the thermal target.

        x:        noisy thermal latents   [B, C, F, H, W]
        cond:     clean RGB latents        [B, C, F, H, W] (zeros for the unconditional branch)
        timestep: [B] in [0, 1000]
        text:     text embeddings          [B, L, text_dim]
        """
        b = x.shape[0]
        tgt, grid = self._patchify(x)
        src, _ = self._patchify(cond)
        tgt = tgt + self.embed_thermal.to(tgt.dtype)
        src = src + self.embed_rgb.to(src.dtype)
        n = tgt.shape[1]
        tokens = torch.cat([src, tgt], 1)
        cos, sin = self.rope(grid, tokens.device)
        rotary = (torch.cat([cos, cos], -2), torch.cat([sin, sin], -2))
        t_all = torch.cat([torch.zeros(b, n, device=tokens.device, dtype=tokens.dtype),
                           timestep.to(tokens.device, tokens.dtype).view(b, 1).expand(b, n)], 1)
        temb, tproj, context = self.condition_embedder(t_all.flatten(), text, tokens.shape[1])
        tproj = tproj.unflatten(2, (6, -1))
        for block in self.blocks:
            tokens = block(tokens, context, tproj, rotary)
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        tokens = (self.norm_out(tokens.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)).type_as(tokens)
        return self._unpatchify(self.proj_out(tokens)[:, -n:], b, grid)
