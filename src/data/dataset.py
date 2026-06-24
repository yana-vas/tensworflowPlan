from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import ImagePreprocessor

# The 13 ShapeNet categories named in the project goal (id -> human name).
CATEGORIES = {
    "02691156": "airplane",
    "02828884": "bench",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03211117": "display",
    "03636649": "lamp",
    "03691459": "speaker",
    "04090263": "rifle",
    "04256520": "sofa",
    "04379243": "table",
    "04401088": "telephone",
    "04530566": "vessel",
}

# OccNet points are stored in [-0.5, 0.5]; we train/infer in [-1, 1].
_POINT_SCALE = 2.0


class ShapeNetDataset(Dataset):
    """ShapeNet image + occupancy dataset with optional surface/camera outputs."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        categories: Optional[List[str]] = None,
        num_points: int = 2048,
        image_size: int = 224,
        augment: bool = False,
        max_samples: Optional[int] = None,
        return_eval: bool = False,
        eval_points: int = 100000,
        load_camera: bool = False,
    ) -> None:
        """Args:
        root: Dataset root containing one subdir per category id.
        split: 'train' | 'val' | 'test'.
        categories: Category ids to include (default: all 13).
        num_points: Volume points sampled per item for training.
        image_size: Image side length (passed to ImagePreprocessor).
        augment: Apply training image augmentation.
        max_samples: Cap total samples (debugging / smoke tests).
        return_eval: Also return surface points+normals from pointcloud.npz.
        eval_points: Surface points to sample when return_eval is True.
        load_camera: Also return camera intrinsics/extrinsics for the chosen view.
        """
        self.root = Path(root)
        self.split = split
        self.categories = categories or list(CATEGORIES.keys())
        self.num_points = num_points
        self.augment = augment
        self.return_eval = return_eval
        self.eval_points = eval_points
        self.load_camera = load_camera
        self.preprocessor = ImagePreprocessor(image_size=image_size)
        self.samples = self._find_samples(max_samples)
        print(f"[{split}] {len(self.samples)} samples across "
              f"{len(set(s['category'] for s in self.samples))} categories")

    def _find_samples(self, max_samples: Optional[int]) -> List[Dict[str, str]]:
        """Scan the root and apply the deterministic 80/10/10 per-category split."""
        samples: List[Dict[str, str]] = []
        for cat_id in sorted(self.categories):
            cat_dir = self.root / cat_id
            if not cat_dir.exists():
                continue
            model_dirs = sorted(
                d for d in cat_dir.iterdir()
                if d.is_dir() and (d / "points.npz").exists()
            )
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(model_dirs))
            n = len(model_dirs)
            train_end, val_end = int(0.8 * n), int(0.9 * n)
            if self.split == "train":
                selected = indices[:train_end]
            elif self.split == "val":
                selected = indices[train_end:val_end]
            else:
                selected = indices[val_end:]
            for idx in selected:
                samples.append({
                    "category": cat_id,
                    "model_id": model_dirs[idx].name,
                    "dir": str(model_dirs[idx]),
                })
                if max_samples and len(samples) >= max_samples:
                    return samples
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, model_dir: Path) -> Tuple[Image.Image, int]:
        """Load one random rendered view; return (image, view_index)."""
        img_dir = model_dir / "img_choy2016"
        if img_dir.exists():
            views = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            if views:
                vi = int(np.random.randint(len(views)))
                return Image.open(str(views[vi])).convert("RGB"), vi
        return Image.new("RGB", (224, 224), "white"), 0

    def _load_points(self, model_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and subsample volume points + binary occupancy, scaled to [-1, 1]."""
        data = np.load(str(model_dir / "points.npz"))
        points = data["points"].astype(np.float32) * _POINT_SCALE
        occ = np.unpackbits(data["occupancies"])[: points.shape[0]].astype(np.float32)
        n = points.shape[0]
        choice = np.random.choice(n, size=self.num_points, replace=n < self.num_points)
        return points[choice], occ[choice]

    def _load_pointcloud(self, model_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load surface points + normals (for evaluation), scaled to [-1, 1]."""
        pc_path = model_dir / "pointcloud.npz"
        if not pc_path.exists():
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty
        data = np.load(str(pc_path))
        pts = data["points"].astype(np.float32) * _POINT_SCALE
        normals = data["normals"].astype(np.float32)
        m = pts.shape[0]
        k = min(self.eval_points, m)
        choice = np.random.choice(m, size=k, replace=False)
        return pts[choice], normals[choice]

    def _load_camera(self, model_dir: Path, view_idx: int) -> Dict[str, np.ndarray]:
        """Load camera world matrix + intrinsics for a view; empty dict if absent."""
        cam_path = model_dir / "img_choy2016" / "cameras.npz"
        if not cam_path.exists():
            return {}
        cam = np.load(str(cam_path))
        return {
            "world_mat": cam[f"world_mat_{view_idx}"].astype(np.float32),
            "camera_mat": cam[f"camera_mat_{view_idx}"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return one sample dict (see module docstring for keys)."""
        info = self.samples[idx]
        model_dir = Path(info["dir"])

        image, view_idx = self._load_image(model_dir)
        image_tensor = self.preprocessor(image, augment=self.augment)
        points, occupancy = self._load_points(model_dir)

        item: Dict[str, object] = {
            "image": image_tensor,
            "points": torch.from_numpy(points),
            "occupancy": torch.from_numpy(occupancy).unsqueeze(-1),
            "category": info["category"],
            "model_id": info["model_id"],
        }
        if self.return_eval:
            pc_pts, pc_normals = self._load_pointcloud(model_dir)
            item["eval_points"] = torch.from_numpy(pc_pts)
            item["eval_normals"] = torch.from_numpy(pc_normals)
        if self.load_camera:
            cam = self._load_camera(model_dir, view_idx)
            for key, value in cam.items():
                item[key] = torch.from_numpy(value)
        return item


def get_dataloader(
    root: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    **kwargs: object,
) -> DataLoader:
    """Build a DataLoader over ShapeNetDataset.

    Shuffles + drops the last partial batch only for training. ``**kwargs`` are
    forwarded to ``ShapeNetDataset`` (num_points, augment, return_eval, ...).
    """
    dataset = ShapeNetDataset(root=root, split=split, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        loader = get_dataloader(sys.argv[1], split="train", batch_size=4, num_workers=0,
                                max_samples=8)
        batch = next(iter(loader))
        print("image", tuple(batch["image"].shape),
              "points", tuple(batch["points"].shape),
              "occ", tuple(batch["occupancy"].shape))
        assert batch["image"].shape[1:] == (3, 224, 224)
        assert batch["points"].shape[-1] == 3
        assert batch["occupancy"].shape[-1] == 1
        print("dataset.py self-test passed")
    else:
        print("Pass a ShapeNet root to self-test: python -m src.data.dataset <DATA_ROOT>")