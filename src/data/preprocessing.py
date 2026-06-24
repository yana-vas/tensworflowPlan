from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ImagePreprocessor:

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        patch_size: int = 14,
    ) -> None:
        
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by ViT patch_size ({patch_size})"
            )
        self.image_size = image_size
        self.mean = mean
        self.std = std

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.transform_augment = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        augment: bool = False,
    ) -> torch.Tensor:
    
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform_augment(image) if augment else self.transform(image)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(self.std, device=tensor.device).view(3, 1, 1)
        return tensor * std + mean


if __name__ == "__main__":
    pre = ImagePreprocessor(image_size=224)
    dummy = Image.new("RGB", (300, 200), "white")
    out = pre(dummy)
    print("preprocessing.py self-test:", tuple(out.shape))
    assert out.shape == (3, 224, 224)