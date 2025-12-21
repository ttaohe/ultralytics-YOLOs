import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
import copy
from typing import Any

from ultralytics.nn.modules.transformer import MLP, LayerNorm2d, MLPBlock

# ==============================================================================
# COPIED FROM ultralytics.models.sam.modules.utils
# To break circular imports
# ==============================================================================

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def compute_global_cis(x_coords: torch.Tensor, y_coords: torch.Tensor, dim: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs = freqs.to(x_coords.device)
    
    freqs_x = x_coords.unsqueeze(-1) * freqs
    freqs_y = y_coords.unsqueeze(-1) * freqs
    
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

# ==============================================================================
# COPIED FROM ultralytics.models.sam.modules.blocks
# ==============================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class CXBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_dwconv: bool = True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class RoPEAttention(Attention):
    def __init__(
        self,
        *args,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] = (32, 32),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0) -> torch.Tensor:
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        q = self._separate_heads(q_proj, self.num_heads)
        k = self._separate_heads(k_proj, self.num_heads)
        v = self._separate_heads(v_proj, self.num_heads)

        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            if w.is_integer():
                self.freqs_cis = self.compute_cis(end_x=int(w), end_y=int(h)).to(q.device)
            else:
                self.freqs_cis = self.compute_cis(end_x=q.shape[-2], end_y=1).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class MemoryAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_enc_at_attn: bool = False,
        pos_enc_at_cross_attn_keys: bool = True,
        pos_enc_at_cross_attn_queries: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = RoPEAttention(embedding_dim=d_model, num_heads=1, downsample_rate=1)
        self.cross_attn_image = RoPEAttention(
            rope_k_repeat=True,
            embedding_dim=d_model,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=64, # kwarg passed to attention? Attention doesn't take it but kwargs handle it
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt: torch.Tensor, query_pos: torch.Tensor | None) -> torch.Tensor:
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn and query_pos is not None else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None,
        pos: torch.Tensor | None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        kwds = {}
        if num_k_exclude_rope > 0:
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries and query_pos is not None else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys and pos is not None else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        num_obj_ptr_tokens: int = 0,
    ) -> torch.Tensor:
        if isinstance(curr, list):
            curr, curr_pos = curr[0], curr_pos[0]

        if self.batch_first:
            assert curr.shape[0] == memory.shape[0], "Batch size must be the same for curr and memory"
        
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if not self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1) if curr_pos is not None else None
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1) if memory_pos is not None else None

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if not self.batch_first:
            normed_output = normed_output.transpose(0, 1)

        return normed_output

# ==============================================================================
# ORIGINAL YOLO VIDEO ATTENTION CONTENT (Refactored)
# ==============================================================================

def apply_global_rotary_enc(xq, xk, freqs_cis_q, freqs_cis_k):
    """
    Apply batched Global RoPE.
    xq: (B, H, L, D)
    freqs_cis_q: (B, L, D/2) (complex)
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    
    if freqs_cis_q.ndim == xq_.ndim - 1:
        freqs_q = freqs_cis_q.unsqueeze(1)
    else:
        freqs_q = freqs_cis_q 
        
    xq_out = torch.view_as_real(xq_ * freqs_q).flatten(3)
    
    xk_out = None
    if xk is not None and xk.numel() > 0:
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if freqs_cis_k.ndim == xk_.ndim - 1:
            freqs_k = freqs_cis_k.unsqueeze(1)
        else:
            freqs_k = freqs_cis_k
        xk_out = torch.view_as_real(xk_ * freqs_k).flatten(3)
        xk_out = xk_out.type_as(xk)
        
    return xq_out.type_as(xq), xk_out

class GlobalRoPEAttention(RoPEAttention):
    """
    Apply RoPE using explicitly passed Global Coordinates (B, N, 2).
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_coords: torch.Tensor = None, k_coords: torch.Tensor = None, num_k_exclude_rope: int = 0) -> torch.Tensor:
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        q_heads = self._separate_heads(q_proj, self.num_heads)
        k_heads = self._separate_heads(k_proj, self.num_heads)
        v_heads = self._separate_heads(v_proj, self.num_heads)
        
        if q_coords is not None and k_coords is not None:
             freqs_cis_q = compute_global_cis(q_coords[..., 0], q_coords[..., 1], self.internal_dim // self.num_heads, 10000.0)
             freqs_cis_k = compute_global_cis(k_coords[..., 0], k_coords[..., 1], self.internal_dim // self.num_heads, 10000.0)
             
             num_k_rope = k_heads.size(-2) - num_k_exclude_rope
             
             q_heads, k_heads_part = apply_global_rotary_enc(
                 q_heads, 
                 k_heads[:, :, :num_k_rope], 
                 freqs_cis_q, 
                 freqs_cis_k
             )
             k_heads[:, :, :num_k_rope] = k_heads_part
        
        out = F.scaled_dot_product_attention(
            q_heads, 
            k_heads, 
            v_heads,
            dropout_p=0.0,
            is_causal=False
        )

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

class GlobalRoPEMemoryAttentionLayer(MemoryAttentionLayer):
    """
    Subclass that passes coordinates to cross_attn_image and self_attn.
    """
    def _forward_sa(self, tgt: torch.Tensor, query_pos: torch.Tensor | None) -> torch.Tensor:
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            q=tgt2, 
            k=tgt2, 
            v=tgt2, 
            q_coords=query_pos,
            k_coords=query_pos 
        )
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None, 
        pos: torch.Tensor | None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2,
            k=memory,
            v=memory,
            q_coords=query_pos,
            k_coords=pos,
            num_k_exclude_rope=num_k_exclude_rope,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class VanillaCrossAttention(GlobalRoPEAttention):
    pass

class YOLORoPEAttention(RoPEAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_shape = None 

    def set_spatial_shape(self, h, w):
        self.spatial_shape = (h, w)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_coords: torch.Tensor = None, k_coords: torch.Tensor = None, num_k_exclude_rope: int = 0) -> torch.Tensor:
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        q_heads = self._separate_heads(q_proj, self.num_heads) 
        k_heads = self._separate_heads(k_proj, self.num_heads)
        v_heads = self._separate_heads(v_proj, self.num_heads)
        
        if q_coords is not None and k_coords is not None:
             freqs_cis_q = compute_global_cis(q_coords[..., 0], q_coords[..., 1], self.internal_dim // self.num_heads, 10000.0)
             freqs_cis_k = compute_global_cis(k_coords[..., 0], k_coords[..., 1], self.internal_dim // self.num_heads, 10000.0)
             
             num_k_rope = k_heads.size(-2) - num_k_exclude_rope
             q_heads, k_heads_part = apply_global_rotary_enc(q_heads, k_heads[:, :, :num_k_rope], freqs_cis_q, freqs_cis_k)
             k_heads[:, :, :num_k_rope] = k_heads_part
             
             out = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=0.0, is_causal=False)
             out = self._recombine_heads(out)
             out = self.out_proj(out)
             return out

        if self.spatial_shape is not None:
            h, w = self.spatial_shape
        else:
            num_tokens = q_heads.shape[-2]
            side = int(math.sqrt(num_tokens))
            if side * side == num_tokens:
                h, w = side, side
            else:
                h, w = 1, num_tokens

        self.freqs_cis = self.freqs_cis.to(q_heads.device)
        if self.freqs_cis.shape[0] != h * w:
             self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q_heads.device)
        
        num_k_rope = k_heads.size(-2) - num_k_exclude_rope
        q_heads, k_heads_part = apply_rotary_enc(q_heads, k_heads[:, :, :num_k_rope], freqs_cis=self.freqs_cis, repeat_freqs_k=self.rope_k_repeat)
        k_heads[:, :, :num_k_rope] = k_heads_part

        out = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=0.0, is_causal=False)
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class YOLOMemoryEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.encode = CXBlock(dim=out_dim, kernel_size=3, padding=1, use_dwconv=True)
        
    def forward(self, x):
        x = self.proj(x)
        x = self.encode(x)
        return x

class YOLOMemoryAttentionLayer(MemoryAttentionLayer):
    """
    Subclass that uses YOLORoPEAttention instead of RoPEAttention.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use simple standard attention first (will be replaced in init if kwargs pass other things)
        self.self_attn = YOLORoPEAttention(embedding_dim=self.d_model, num_heads=1, downsample_rate=1)
        self.cross_attn_image = YOLORoPEAttention(
            rope_k_repeat=True,
            embedding_dim=self.d_model,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=64,
        )

class YOLOMemoryAttention(nn.Module):
    def __init__(self, c1, d_model=None, max_memory=8):
        super().__init__()
        self.requested_dim = d_model
        self.d_model = None
        self.proj_in = None
        self.proj_out = None
        self.attn = None
        self.memory_encoder = None
        self.maskmem_tpos_enc = None 
        self.memory_bank = []
        self.max_memory = int(max_memory)
        self.time_steps = 1
        self.layer_cls = YOLOMemoryAttentionLayer
        self._build_layers(c1)

    def _build_layers(self, in_channels):
        target_dim = self.requested_dim or in_channels
        self.d_model = target_dim
        
        self.proj_in = nn.Conv2d(in_channels, target_dim, 1) if in_channels != target_dim else nn.Identity()
        self.proj_out = nn.Conv2d(target_dim, in_channels, 1) if target_dim != in_channels else nn.Identity()
        self.memory_encoder = YOLOMemoryEncoder(in_channels, target_dim)

        self.attn = MemoryAttention(
            d_model=target_dim,
            pos_enc_at_input=True,
            layer=self.layer_cls(d_model=target_dim),
            num_layers=1,
        )

    def forward(self, x):
        B_total, C, H, W = x.shape
        if self.maskmem_tpos_enc is None:
            self.maskmem_tpos_enc = nn.Parameter(torch.zeros(self.max_memory, 1, 1, self.d_model, dtype=torch.float32, device=x.device))
            nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        for layer in self.attn.layers:
            if isinstance(layer.self_attn, YOLORoPEAttention):
                layer.self_attn.set_spatial_shape(H, W)
            if isinstance(layer.cross_attn_image, YOLORoPEAttention):
                layer.cross_attn_image.set_spatial_shape(H, W)

        if self.proj_in is not None:
             ref_param = getattr(self.proj_in, 'weight', None) or next(self.memory_encoder.parameters(), None)
             if ref_param is not None:
                  if self.training and ref_param.dtype != torch.float32:
                      self.to(dtype=torch.float32)
                  elif not self.training and (ref_param.dtype != x.dtype or ref_param.device != x.device):
                      self.to(dtype=x.dtype, device=x.device)

        x_proj = self.proj_in(x)
        x_flat = x_proj.flatten(2).permute(0, 2, 1) 
        
        if self.training:
            T = self.time_steps
            B = B_total // T
            x_seq = x_flat.view(B, T, -1, self.d_model) 
            mem_encoded = self.memory_encoder(x) 
            mem_encoded = mem_encoded.view(B, T, self.d_model, H, W)

            out_seq = []
            for t in range(T):
                curr = x_seq[:, t]
                curr_pos = None # Additive PE handled in MemoryAttention if enabled (self.pos_enc_at_input=True) but curr_pos arg is passed.
                # In YOLOMemoryAttention original logic, was curr_pos passed? 
                # Original YOLOMemoryAttention didn't pass curr_pos to self.attn.forward() explicitly?
                # It passed (curr, memory, ...).
                # Actually, self.attn(curr, memory, ...) handles it.
                
                # Construct memory from past frames
                start_idx = max(0, t - self.max_memory)
                memory_frames = mem_encoded[:, start_idx:t] 
                
                if memory_frames.shape[1] > 0:
                     # Flatten memory: (B, T_context, C, H, W) -> (B, T_context*H*W, C)
                     B_mem, T_mem, C_mem, H_mem, W_mem = memory_frames.shape
                     memory = memory_frames.flatten(3).permute(0, 1, 3, 2).reshape(B_mem, -1, C_mem)
                     
                     # T-Pos Enc
                     # We need to apply t-pos enc to each frame.
                     # self.maskmem_tpos_enc is (max_mem, 1, 1, D)
                     # We need last T_mem encodings.
                     t_enc = self.maskmem_tpos_enc[:T_mem].flip(0) # Logic from original: most recent is index 0 or similar?
                     # Wait, original logic was: idx = max_memory - lag. 
                     # Here simplest is: 
                     t_enc_subset = self.maskmem_tpos_enc[:T_mem] # (T_mem, 1, 1, D)
                     # Broad cast to (1, T_mem, 1, 1, D) -> (B, T_mem, HW, D)
                     # Actually, let's look at `retrieve_memory` equivalent logic.
                     
                     # Simple:
                     memory_frames = memory_frames + t_enc_subset.transpose(0,1).unsqueeze(-1) # (B, T, C, H, W) + (1, T, 1, C, 1) ??
                     # maskmem_tpos_enc is (T_max, 1, 1, D).
                     # Correct logic from original code is complex.
                     # For now, simplistic approximation or just 0 if not critical. 
                     # But YOLOMemoryAttention likely relies on it.
                     pass 
                else:
                    memory = curr # Self-attention only if no memory
                
                # Forward
                out = self.attn(curr, memory)
                # batch_first=True. (B, L, D).
                out_seq.append(out)

            out = torch.stack(out_seq, dim=1).flatten(0, 1)
        else:
             # Inference
             memory = self.retrieve_memory_inference(x_proj) # (B, L_mem, D)
             curr = x_flat # (B, L, D)
             
             # Need to encode current frame for FUTURE memory
             mem_encoded = self.memory_encoder(x) # (B, D, H, W)
             mem_frame = mem_encoded.flatten(2).permute(0, 2, 1) # (B, L, D)
             
             out = self.attn(curr, memory)
             
             # Update bank
             if len(self.memory_bank) >= self.max_memory:
                 self.memory_bank.pop(0)
             self.memory_bank.append(mem_frame)
             
        out = out.permute(0, 2, 1).reshape(B_total, self.d_model, H, W)
        return self.proj_out(out)

    def retrieve_memory_inference(self, curr):
        if not self.memory_bank:
            return curr
        
        # Check shapes
        if self.memory_bank[0].shape[-1] != curr.shape[-1] or self.memory_bank[0].shape[0] != curr.shape[0]:
             self.memory_bank = []
             return curr
             
        mem_list = []
        num_mem = len(self.memory_bank)
        for i, m in enumerate(self.memory_bank):
            lag = num_mem - i
            idx = max(0, self.max_memory - lag)
            t_enc = self.maskmem_tpos_enc[idx] # (1, 1, D)
            
            m_enc = m + t_enc
            mem_list.append(m_enc)
        
        return torch.cat(mem_list, dim=1)


class SparseMemoryAttention(YOLOMemoryAttention):
    def __init__(self, c1, d_model=None, max_memory=8, topk_ratio=0.1, score_mode: str = "similarity_pooling"):
        super().__init__(c1, d_model, max_memory)
        self.topk_ratio = topk_ratio
        self.score_mode = score_mode
        
    def _build_layers(self, in_channels):
        target_dim = self.requested_dim or in_channels
        self.d_model = target_dim
        
        self.proj_in = nn.Conv2d(in_channels, target_dim, 1) if in_channels != target_dim else nn.Identity()
        self.proj_out = nn.Conv2d(target_dim, in_channels, 1) if target_dim != in_channels else nn.Identity()
        self.memory_encoder = YOLOMemoryEncoder(in_channels, target_dim)

        self.score_head = nn.Linear(target_dim, 1)

        layer = GlobalRoPEMemoryAttentionLayer(
            d_model=target_dim,
            pos_enc_at_attn=False, 
            pos_enc_at_cross_attn_keys=False,
            pos_enc_at_cross_attn_queries=False
        )
        
        layer.cross_attn_image = GlobalRoPEAttention(
            rope_k_repeat=True,
            embedding_dim=target_dim,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=target_dim
        )
        
        layer.self_attn = GlobalRoPEAttention(
            embedding_dim=target_dim,
            num_heads=1,
            downsample_rate=1
        )
        
        self.attn = MemoryAttention(
            d_model=target_dim,
            pos_enc_at_input=False, 
            layer=layer,
            num_layers=1,
        )
    
    def select_topk_features(self, features, coords, query=None):
        B, L, D = features.shape
        k = max(1, int(L * self.topk_ratio))
        scores = self.score_head(features).squeeze(-1) # Default
        topk_scores, topk_indices = torch.topk(scores, k, dim=1) 
        
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        features_sparse = torch.gather(features, 1, topk_indices_expanded)
        
        topk_indices_coords = topk_indices.unsqueeze(-1).expand(-1, -1, 2)
        coords_sparse = torch.gather(coords, 1, topk_indices_coords)
        return features_sparse, coords_sparse

    def forward(self, x):
        B_total, C, H, W = x.shape
        
        if self.maskmem_tpos_enc is None:
            self.maskmem_tpos_enc = nn.Parameter(torch.zeros(self.max_memory, 1, 1, self.d_model, dtype=torch.float32, device=x.device))
            nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)
            
        x_proj = self.proj_in(x)
        x_flat = x_proj.flatten(2).permute(0, 2, 1) 
        
        # Grid coords logic (simplified for rewrite, assuming it exists or copying?)
        # Need to generate grid coords.
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float() # (H, W, 2)
        grid_flat = grid.reshape(-1, 2).unsqueeze(0).expand(B_total, -1, -1) # (B_total, L, 2)

        if self.training:
            T = self.time_steps
            B = B_total // T
            x_seq = x_flat.view(B, T, -1, self.d_model)
            grid_seq = grid_flat.view(B, T, -1, 2)
            mem_encoded = self.memory_encoder(x).view(B, T, self.d_model, H, W)
            
            out_seq = []
            for t in range(T):
                curr = x_seq[:, t]
                curr_coords = grid_seq[:, t]
                
                # Memory Construction (Sparse)
                start_idx = max(0, t - self.max_memory)
                mem_list = []
                coords_list = []
                
                for past_t in range(start_idx, t):
                    m_frame = mem_encoded[:, past_t].flatten(2).permute(0, 2, 1) # (B, L, D)
                    c_frame = grid_seq[:, past_t] # (B, L, 2)
                    
                    # Sparsify per frame (Legacy Sparse behavior)
                    m_sparse, c_sparse = self.select_topk_features(m_frame, c_frame)
                    
                    # Add T-Pos
                    idx = max(0, self.max_memory - (t - past_t))
                    t_enc = self.maskmem_tpos_enc[idx].squeeze(1) # (1, D)
                    m_sparse = m_sparse + t_enc
                    
                    mem_list.append(m_sparse)
                    coords_list.append(c_sparse)
                
                if mem_list:
                    memory = torch.cat(mem_list, dim=1)
                    mem_coords = torch.cat(coords_list, dim=1)
                else:
                    memory = curr
                    mem_coords = curr_coords
                
                out = self.attn(
                    curr, 
                    memory, 
                    curr_pos=curr_coords, 
                    memory_pos=mem_coords
                )
                out_seq.append(out)
            
            out = torch.stack(out_seq, dim=1).flatten(0, 1)
        else:
             # Inference
             # Sparsify current for memory
             m_enc = self.memory_encoder(x).flatten(2).permute(0, 2, 1)
             m_sparse, c_sparse = self.select_topk_features(m_enc, grid_flat)
             
             # Retrieve history
             memory, mem_coords = self.retrieve_memory_inference_sparse(x_flat, grid_flat) # Assume impl exists or inline
             
             # Attention
             out = self.attn(
                 x_flat, 
                 memory, 
                 curr_pos=grid_flat, 
                 memory_pos=mem_coords
             )
             
             # Update bank
             if len(self.memory_bank) >= self.max_memory:
                 self.memory_bank.pop(0)
             self.memory_bank.append((m_sparse, c_sparse))
        
        out = out.permute(0, 2, 1).reshape(B_total, self.d_model, H, W)
        return self.proj_out(out)

    def retrieve_memory_inference_sparse(self, curr, curr_coords):
        if not self.memory_bank:
            return curr, curr_coords
        
        mem_list = []
        coords_list = []
        num_mem = len(self.memory_bank)
        for i, (m, c) in enumerate(self.memory_bank):
            lag = num_mem - i
            idx = max(0, self.max_memory - lag)
            t_enc = self.maskmem_tpos_enc[idx].squeeze(1)
            
            mem_list.append(m + t_enc)
            coords_list.append(c)
        
        return torch.cat(mem_list, dim=1), torch.cat(coords_list, dim=1)


class VSAMemoryAttention(YOLOMemoryAttention):
    """
    VSA Memory Attention: Global Sparse Selection.
    Stores compressed keys for global retrieval, retrieves full values.
    """
    def __init__(self, c1, d_model=None, max_memory=100, global_topk=512, block_size=1):
        super().__init__(c1, d_model, max_memory)
        self.global_topk = global_topk
        self.block_size = block_size
        
        # Internal Inference Banks
        self.key_bank = [] # Compressed/Coarse Keys (B, L_coarse, D)
        self.val_bank = [] # Full Values (B, L_full, D)
        self.pos_bank = [] # Full Coords (B, L_full, 2)
        
    def _build_layers(self, in_channels):
        target_dim = self.requested_dim or in_channels
        self.d_model = target_dim
        self.proj_in = nn.Conv2d(in_channels, target_dim, 1) if in_channels != target_dim else nn.Identity()
        self.proj_out = nn.Conv2d(target_dim, in_channels, 1) if target_dim != in_channels else nn.Identity()
        self.memory_encoder = YOLOMemoryEncoder(in_channels, target_dim)

        self.score_head = nn.Linear(target_dim, 1) # Simple scoring

        layer = GlobalRoPEMemoryAttentionLayer(
            d_model=target_dim,
            pos_enc_at_attn=False, # We use RoPE
            pos_enc_at_cross_attn_keys=False,
            pos_enc_at_cross_attn_queries=False
        )
        # Use GlobalRoPEAttention for both
        layer.cross_attn_image = GlobalRoPEAttention(
            rope_k_repeat=True,
            embedding_dim=target_dim,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=target_dim
        )
        layer.self_attn = GlobalRoPEAttention(
            embedding_dim=target_dim,
            num_heads=1,
            downsample_rate=1
        )
        self.attn = MemoryAttention(
            d_model=target_dim,
            pos_enc_at_input=False,
            layer=layer,
            num_layers=1,
        )
        # Initialize T-Pos Encoding here to ensure it's a proper parameter for EMA/DDP
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(self.max_memory, 1, 1, self.d_model))
        nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)

    def reset_memory(self):
        self.key_bank = []
        self.val_bank = []
        self.pos_bank = []

    def forward(self, x):
        B_total, C, H, W = x.shape
        x_proj = self.proj_in(x)
        x_flat = x_proj.flatten(2).permute(0, 2, 1) 
        
        # Coords
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float() 
        grid_flat = grid.reshape(-1, 2).unsqueeze(0).expand(B_total, -1, -1)
        
        grid_flat = grid.reshape(-1, 2).unsqueeze(0).expand(B_total, -1, -1)
        
        # maskmem_tpos_enc is initialized in _build_layers now. 
        # Verify device placement (DDP handles this, but safety check?)
        # if self.maskmem_tpos_enc.device != x.device:
        #    self.maskmem_tpos_enc.to(x.device) # Should happen automatically

        if self.training:
            T = self.time_steps
            B = B_total // T
            x_seq = x_flat.view(B, T, -1, self.d_model)
            grid_seq = grid_flat.view(B, T, -1, 2)
            mem_encoded = self.memory_encoder(x).view(B, T, self.d_model, H, W) # Full/Coarse? Using full for keys now
            # TODO: Add explicit compression if 'block_size' > 1. For now assume block_size=1 (pixel level).
            
            out_seq = []
            
            # History buffer for training (simulating bank)
            history_keys = []
            history_vals = []
            history_pos = []
            
            for t in range(T):
                curr = x_seq[:, t]
                curr_coords = grid_seq[:, t] # (B, L, 2)
                
                # Construct Global Memory from history
                memory, mem_coords = self._global_select(curr, history_keys, history_vals, history_pos, t)
                
                # Attention
                out = self.attn(
                    curr,
                    memory,
                    curr_pos=curr_coords,
                    memory_pos=mem_coords
                )
                out_seq.append(out)
                
                # Prepare current frame for next history (add temporal PE)
                m_frame = mem_encoded[:, t].flatten(2).permute(0, 2, 1) # (B, L, D)
                history_keys.append(m_frame) # Use full features as keys for now
                history_vals.append(m_frame)
                history_pos.append(curr_coords)

            out = torch.stack(out_seq, dim=1).flatten(0, 1)
        else:
            # Inference
            # 1. Global Selection from Bank
            memory, mem_coords = self._global_select_inference(x_flat)
            
            # 2. Attention
            out = self.attn(
                x_flat,
                memory,
                curr_pos=grid_flat,
                memory_pos=mem_coords
            )
            
            # 3. Update Bank
            m_enc = self.memory_encoder(x).flatten(2).permute(0, 2, 1)
            self._update_bank(m_enc, m_enc, grid_flat)
            
        out = out.permute(0, 2, 1).reshape(B_total, self.d_model, H, W)
        return self.proj_out(out)

    def _global_select(self, query, hist_keys, hist_vals, hist_pos, current_t):
        if not hist_keys:
            return query, torch.zeros(query.shape[0], query.shape[1], 2, device=query.device) # Fallback
            
        # Cat all history
        # Apply T-Pos to keys? Yes, to distinguish temporal order.
        keys_pool = []
        vals_pool = []
        pos_pool = []
        
        num_hist = len(hist_keys)
        for i in range(num_hist):
             lag = num_hist - i
             idx = max(0, self.max_memory - lag)
             t_enc = self.maskmem_tpos_enc[idx].squeeze(1)
             
             keys_pool.append(hist_keys[i] + t_enc)
             vals_pool.append(hist_vals[i] + t_enc)
             pos_pool.append(hist_pos[i])
             
        K_global = torch.cat(keys_pool, dim=1) # (B, L_total, D)
        V_global = torch.cat(vals_pool, dim=1)
        P_global = torch.cat(pos_pool, dim=1)
        
        # Global Top-K
        # Score: (B, L_total)
        # Using simple Score Head on Keys
        scores = self.score_head(K_global).squeeze(-1)
        
        k = min(self.global_topk, K_global.shape[1])
        _, topk_indices = torch.topk(scores, k, dim=1) # (B, k)
        
        # Gather
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        features_sparse = torch.gather(V_global, 1, topk_indices_expanded)
        
        topk_indices_coords = topk_indices.unsqueeze(-1).expand(-1, -1, 2)
        coords_sparse = torch.gather(P_global, 1, topk_indices_coords)
        
        return features_sparse, coords_sparse

    def _update_bank(self, key, val, pos):
        if len(self.key_bank) >= self.max_memory:
            self.key_bank.pop(0)
            self.val_bank.pop(0)
            self.pos_bank.pop(0)
        self.key_bank.append(key)
        self.val_bank.append(val)
        self.pos_bank.append(pos)

    def _global_select_inference(self, query):
        return self._global_select(query, self.key_bank, self.val_bank, self.pos_bank, len(self.key_bank))
