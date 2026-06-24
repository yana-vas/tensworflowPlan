import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionalEncoding(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        num_bands: int = 10,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_bands = num_bands
        self.include_input = include_input

        freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=True)

        self.output_dim = (in_dim if include_input else 0) + in_dim * 2 * num_bands

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        scaled = coords.unsqueeze(-1) * self.freqs
        
        sincos = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        
        encoded = sincos.flatten(start_dim=-2)
        
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
            
        return encoded


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, nq, _ = query.shape
        nk = kv.shape[1]

        q = self.q_proj(query).view(b, nq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(kv).view(b, nk, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(kv).view(b, nk, self.n_heads, self.d_k).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )

        attn = attn.transpose(1, 2).contiguous().view(b, nq, self.d_model)
        return self.out_proj(attn)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult, dropout)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_q(x), self.norm_kv(kv))
        
        # Feedforward with residual connection: x = x + ffn(norm(x))
        x = x + self.ffn(self.norm_ff(x))
        
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Testing layers.py layout...")
    
    pe = FourierPositionalEncoding(in_dim=3, num_bands=10)
    pts = torch.rand(2, 128, 3) * 2 - 1
    enc = pe(pts)
    print(f"  Fourier encoding output shape: {enc.shape} (Expected: (2, 128, 63))")
    assert enc.shape == (2, 128, 63)

    block = CrossAttentionBlock(d_model=384, n_heads=6, dropout=0.0)
    q = torch.randn(2, 128, 384)
    kv = torch.randn(2, 257, 384)
    out = block(q, kv)
    print(f"  CrossAttentionBlock output shape: {out.shape} (Expected: (2, 128, 384))")
    assert out.shape == (2, 128, 384)
    print("Self-test passed!")