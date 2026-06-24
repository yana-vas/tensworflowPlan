import torch
import torch.nn as nn

_DINOV2_EMBED_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}

class DINOv2Encoder(nn.Module):
    def __init__(
        self,
        variant: str = "dinov2_vits14",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if variant not in _DINOV2_EMBED_DIM:
            raise ValueError(f"Unknown DINOv2 variant {variant}")
            
        self.variant = variant
        self.embed_dim = _DINOV2_EMBED_DIM[variant]
        self.patch_size = 14

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", variant, trust_repo=True
        )

        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] % self.patch_size != 0 or images.shape[-2] % self.patch_size != 0:
            raise ValueError(
                f"Image size must be divisible by patch size {self.patch_size}"
            )
            
        feats = self.backbone.forward_features(images)
        cls = feats["x_norm_clstoken"].unsqueeze(1)
        patches = feats["x_norm_patchtokens"]
        return torch.cat([cls, patches], dim=1)

    def get_num_params(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad or not trainable_only
        )


if __name__ == "__main__":
    print("Testing encoder.py setup...")
    enc = DINOv2Encoder("dinov2_vits14", freeze=True)
    enc.eval()
    with torch.no_grad():
        tokens = enc(torch.randn(2, 3, 224, 224))
    print(f"  Output token shape: {tokens.shape} (Expected: (2, 257, 384))")
    assert tokens.shape == (2, 257, 384)
    assert enc.get_num_params(trainable_only=True) == 0
    print("Self-test passed!")