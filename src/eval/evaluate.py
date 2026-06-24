import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import trimesh

from src.data import CATEGORIES, get_dataloader
from src.eval.metrics import chamfer_l1, f_score, normal_consistency, volumetric_iou
from src.mesh import extract_mesh
from src.model import OccupancyNetwork
from src.utils.config import load_config


def _sample_mesh_surface(mesh: trimesh.Trimesh, n: int) -> tuple:
    points, face_idx = trimesh.sample.sample_surface(mesh, n)
    normals = mesh.face_normals[face_idx]
    return points.astype(np.float32), normals.astype(np.float32)


@torch.no_grad()
def evaluate(
    model: OccupancyNetwork,
    data_root: str,
    device: torch.device,
    resolution: int = 64,
    threshold: float = 0.5,
    max_samples: int = None,
    surface_samples: int = 50000,
    f_tau: float = 0.02,
) -> Dict[str, Dict[str, float]]:
    
    model.eval()
    loader = get_dataloader(
        root=data_root, split="test", batch_size=1, num_workers=0,
        max_samples=max_samples, return_eval=True, augment=False,
    )
    per_cat: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for batch in loader:
        image = batch["image"].to(device)             # (1, 3, 224, 224)
        points = batch["points"].to(device)           # (1, N, 3)
        gt_occ = batch["occupancy"].squeeze(0).squeeze(-1).cpu().numpy()  # (N,)
        cat = CATEGORIES.get(batch["category"][0], batch["category"][0])

        logits = model(image, points)                 # (1, N, 1)
        pred_prob = torch.sigmoid(logits).squeeze(0).squeeze(-1).cpu().numpy()
        iou = volumetric_iou(pred_prob, gt_occ, threshold)

        grid = model.generate_occupancy_grid(image, resolution=resolution)
        mesh = extract_mesh(grid, threshold=threshold, postprocess=True)

        gt_pts = batch["eval_points"].squeeze(0).cpu().numpy()
        gt_normals = batch["eval_normals"].squeeze(0).cpu().numpy()

        if mesh is not None and len(mesh.faces) > 0 and len(gt_pts) > 0:
            pred_pts, pred_normals = _sample_mesh_surface(mesh, surface_samples)
            cd = chamfer_l1(pred_pts, gt_pts)
            nc = normal_consistency(pred_pts, pred_normals, gt_pts, gt_normals)
            fs = f_score(pred_pts, gt_pts, tau=f_tau)
        else:
            cd, nc, fs = float("inf"), 0.0, 0.0

        per_cat[cat].append({"iou": iou, "chamfer": cd, "nc": nc, "fscore": fs})

    results: Dict[str, Dict[str, float]] = {}
    all_rows: List[Dict[str, float]] = []
    for cat, rows in per_cat.items():
        results[cat] = _aggregate(rows)
        all_rows.extend(rows)
    results["mean"] = _aggregate(all_rows)
    return results


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"iou": 0.0, "chamfer": float("inf"), "nc": 0.0, "fscore": 0.0, "n": 0}
    finite_cd = [r["chamfer"] for r in rows if np.isfinite(r["chamfer"])]
    return {
        "iou": float(np.mean([r["iou"] for r in rows])),
        "chamfer": float(np.mean(finite_cd)) if finite_cd else float("inf"),
        "nc": float(np.mean([r["nc"] for r in rows])),
        "fscore": float(np.mean([r["fscore"] for r in rows])),
        "n": len(rows),
    }


def _print_table(results: Dict[str, Dict[str, float]]) -> None:
    header = f"{'category':<12}{'n':>5}{'IoU':>9}{'Chamfer':>11}{'NC':>9}{'F-score':>9}"
    print(header)
    print("-" * len(header))
    for cat in sorted(k for k in results if k != "mean"):
        r = results[cat]
        print(f"{cat:<12}{r['n']:>5}{r['iou']:>9.4f}{r['chamfer']:>11.4f}"
              f"{r['nc']:>9.4f}{r['fscore']:>9.4f}")
    r = results["mean"]
    print("-" * len(header))
    print(f"{'MEAN':<12}{r['n']:>5}{r['iou']:>9.4f}{r['chamfer']:>11.4f}"
          f"{r['nc']:>9.4f}{r['fscore']:>9.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a 3DScan checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OccupancyNetwork.from_checkpoint(args.checkpoint, map_location=str(device)).to(device)

    resolution = args.resolution or config.inference.grid_resolution
    threshold = args.threshold or config.inference.threshold

    results = evaluate(
        model, args.data_root, device,
        resolution=resolution, threshold=threshold, max_samples=args.max_samples,
    )
    _print_table(results)


if __name__ == "__main__":
    main()