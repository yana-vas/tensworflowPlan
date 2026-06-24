import torch
import torch.nn as nn
from src.model.layers import CrossAttentionBlock, FourierPositionalEncoding


class CrossAttentionOccupancyDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int = 384,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 4,
        num_bands: int = 10,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model

        self.pos_enc = FourierPositionalEncoding(in_dim=3, num_bands=num_bands)
        self.query_proj = nn.Linear(self.pos_enc.output_dim, d_model)
        self.token_proj = nn.Linear(token_dim, d_model)

        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model, n_heads, dropout, ffn_mult)
            for _ in range(n_layers)
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, points: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(self.pos_enc(points))
        kv = self.token_proj(tokens)

        for block in self.blocks:
            q = block(q, kv)

        q = self.final_norm(q)
        return self.head(q)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing decoder.py setup...")
    dec = CrossAttentionOccupancyDecoder(token_dim=384, d_model=384, n_heads=6, n_layers=4)
    pts = torch.rand(2, 2048, 3) * 2 - 1
    tok = torch.randn(2, 257, 384)
    logits = dec(pts, tok)
    print(f"  Output logits shape: {logits.shape} (Expected: (2, 2048, 1))")
    assert logits.shape == (2, 2048, 1)
    print("Self-test passed!")