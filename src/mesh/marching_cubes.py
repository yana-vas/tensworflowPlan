"""Marching Cubes: probability grid -> triangle mesh, with edge-closing padding."""

from typing import Optional, Tuple

import numpy as np
import trimesh
from skimage import measure


class MarchingCubesExtractor:
    """Extract an iso-surface mesh from an occupancy probability grid."""

    def __init__(self, threshold: float = 0.5, pad: bool = True) -> None:
        """Args:
        threshold: Iso-level (probability) defining the surface.
        pad: Zero-pad the grid by 1 voxel so edge-touching surfaces close.
        """
        self.threshold = threshold
        self.pad = pad

    def extract(
        self,
        occupancy_grid: np.ndarray,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        postprocess: bool = True,
    ) -> Optional[trimesh.Trimesh]:
        """Run Marching Cubes and map vertices into world bounds.

        Args:
            occupancy_grid: ``(R, R, R)`` probabilities in [0, 1].
            bounds: (low, high) world coordinate range the grid spans.
            postprocess: If True, run ``make_printable`` on the result.

        Returns:
            A ``trimesh.Trimesh`` or None if no surface crosses the threshold.
        """
        grid = occupancy_grid
        if self.pad:
            grid = np.pad(grid, pad_width=1, mode="constant", constant_values=0.0)

        if grid.max() < self.threshold or grid.min() > self.threshold:
            print("Warning: no surface at this threshold (grid never crosses level)")
            return None

        try:
            verts, faces, normals, _ = measure.marching_cubes(grid, level=self.threshold)
        except (ValueError, RuntimeError) as exc:
            print(f"Marching cubes failed: {exc}")
            return None

        # Map voxel indices -> [0,1] -> world bounds. Account for the padding
        # offset so geometry stays aligned with the original [-1,1] frame.
        res = grid.shape[0]
        if self.pad:
            verts = verts - 1.0   # undo the 1-voxel pad shift
            res = res - 2         # original resolution
        verts = verts / max(res - 1, 1)
        verts = verts * (bounds[1] - bounds[0]) + bounds[0]

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.fix_normals()

        if postprocess:
            # Преместен импорт тук, за да се избегне циклична зависимост (circular import)
            from src.mesh.postprocess import make_printable
            mesh = make_printable(mesh)
        return mesh


def extract_mesh(
    occupancy_grid: np.ndarray,
    threshold: float = 0.5,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    pad: bool = True,
    postprocess: bool = True,
) -> Optional[trimesh.Trimesh]:
    """Convenience wrapper around ``MarchingCubesExtractor``."""
    return MarchingCubesExtractor(threshold=threshold, pad=pad).extract(
        occupancy_grid, bounds=bounds, postprocess=postprocess
    )


if __name__ == "__main__":
    # A solid sphere occupancy field reaching the grid edge: padding must close it.
    R = 48
    ax = np.linspace(-1, 1, R)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    sphere = (np.sqrt(gx**2 + gy**2 + gz**2) < 0.9).astype(np.float32)
    mesh = extract_mesh(sphere, threshold=0.5)
    print("marching_cubes.py self-test:",
          "V", len(mesh.vertices), "F", len(mesh.faces), "watertight", mesh.is_watertight)
    assert mesh is not None and mesh.is_watertight