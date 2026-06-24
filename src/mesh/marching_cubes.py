
from typing import Optional, Tuple

import numpy as np
import trimesh
from skimage import measure


class MarchingCubesExtractor:

    def __init__(self, threshold: float = 0.5, pad: bool = True) -> None:
        
        self.threshold = threshold
        self.pad = pad

    def extract(
        self,
        occupancy_grid: np.ndarray,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        postprocess: bool = True,
    ) -> Optional[trimesh.Trimesh]:
        
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

        res = grid.shape[0]
        if self.pad:
            verts = verts - 1.0 
            res = res - 2 # original resolution
        verts = verts / max(res - 1, 1)
        verts = verts * (bounds[1] - bounds[0]) + bounds[0]

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.fix_normals()

        if postprocess:
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
    return MarchingCubesExtractor(threshold=threshold, pad=pad).extract(
        occupancy_grid, bounds=bounds, postprocess=postprocess
    )


if __name__ == "__main__":
    R = 48
    ax = np.linspace(-1, 1, R)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    sphere = (np.sqrt(gx**2 + gy**2 + gz**2) < 0.9).astype(np.float32)
    mesh = extract_mesh(sphere, threshold=0.5)
    print("marching_cubes.py self-test:",
          "V", len(mesh.vertices), "F", len(mesh.faces), "watertight", mesh.is_watertight)
    assert mesh is not None and mesh.is_watertight