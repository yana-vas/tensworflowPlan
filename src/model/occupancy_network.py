from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
from src.model.decoder import CrossAttentionOccupancyDecoder
from src.model.encoder import DINOv2Encoder


class OccupancyNetwork(nn.Module):
    def __init__(
        self,
        encoder_variant: str = "dinov2_vits14",
        freeze_encoder: bool = True,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 4,
        num_bands: int = 10,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.config_snapshot = {
            "encoder_variant": encoder_variant,
            "freeze_encoder": freeze_encoder,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "num_bands": num_bands,
            "dropout": dropout,
            "ffn_mult": ffn_mult,
        }
        self.encoder = DINOv2Encoder(
            variant=encoder_variant,
            freeze=freeze_encoder,
        )
        self.decoder = CrossAttentionOccupancyDecoder(
            token_dim=self.encoder.embed_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            num_bands=num_bands,
            dropout=dropout,
            ffn_mult=ffn_mult,
        )

    @classmethod
    def from_config(cls, config: Any) -> "OccupancyNetwork":
        enc = config.model.encoder
        dec = config.model.decoder
        return cls(
            encoder_variant=enc.variant,
            freeze_encoder=enc.freeze,
            d_model=dec.d_model,
            n_heads=dec.n_heads,
            n_layers=dec.n_layers,
            num_bands=dec.num_bands,
            dropout=dec.dropout,
            ffn_mult=dec.ffn_mult,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, images: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(images)
        return self.decoder(points, tokens)

    @torch.no_grad()
    def generate_occupancy_grid(
        self,
        image: torch.Tensor,
        resolution: int = 64,
        query_batch_size: int = 100000,
        bounds: tuple = (-1.0, 1.0),
    ) -> np.ndarray:
        if image.shape[0] != 1:
            raise ValueError(f"generate_occupancy_grid expects batch size 1, got {image.shape[0]}")
            
        device = next(self.parameters()).device
        self.eval()

        tokens = self.encode(image.to(device))

        axis = torch.linspace(bounds[0], bounds[1], resolution, device=device)
        gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
        points = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

        probs = torch.empty(points.shape[0], device=device)
        for start in range(0, points.shape[0], query_batch_size):
            chunk = points[start:start + query_batch_size].unsqueeze(0)
            logits = self.decoder(chunk, tokens)
            probs[start:start + chunk.shape[1]] = torch.sigmoid(logits).squeeze(0).squeeze(-1)

        return probs.reshape(resolution, resolution, resolution).cpu().numpy()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.config_snapshot}, path)

    @classmethod
    def from_checkpoint(cls, path: str, map_location: str = "cpu") -> "OccupancyNetwork":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    def get_num_params(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad or not trainable_only
        )


if __name__ == "__main__":
    print("Testing occupancy_network.py setup...")
    model = OccupancyNetwork()
    model.eval()
    img = torch.randn(2, 3, 224, 224)
    pts = torch.rand(2, 512, 3) * 2 - 1
    with torch.no_grad():
        out = model(img, pts)
    print(f"  Forward pass output shape: {out.shape} (Expected: (2, 512, 1))")
    assert out.shape == (2, 512, 1)

    grid = model.generate_occupancy_grid(img[:1], resolution=16, query_batch_size=1024)
    print(f"  Generated grid shape: {grid.shape} (Expected: (16, 16, 16))")
    assert grid.shape == (16, 16, 16)
    print("Self-test passed!")