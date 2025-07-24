from __future__ import annotations
from typing import Optional
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    print("✅ xFormers found. Using memory-efficient attention.")
except ImportError:
    XFORMERS_AVAILABLE = False
    print("⚠️ xFormers not found. Using standard PyTorch attention.")


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


class Attention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            head_dim: int,
            dropout: float = 0.0,
        ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        self.qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(self.num_heads * self.head_dim, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        if XFORMERS_AVAILABLE:
            attn_bias = xops.fmha.attn_bias.LowerTriangularMask() if causal else None
            out = xops.memory_efficient_attention(
                q, k, v, 
                p=self.dropout if self.training else 0.0,
                attn_bias=attn_bias,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / self.head_dim**0.5)
            if causal:
                mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
                att = att.masked_fill(mask == 0, float("-inf"))
            
            att = F.softmax(att, dim=-1)
            if self.training:
                att = F.dropout(att, p=self.dropout)
            
            out = att @ v
        
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)
        

class CausalAttention(Attention):
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        return super().forward(x, causal=causal)
