"""Post-process a raw mesh into a watertight, printable solid."""

from typing import Optional

import numpy as np
import trimesh


def make_printable(
    mesh: trimesh.Trimesh,
    keep_largest: bool = True,
    fill_holes: bool = True,
) -> Optional[trimesh.Trimesh]:
    """Clean a Marching-Cubes mesh for 3D printing.

    Steps: drop degenerate faces, optionally keep only the largest connected
    component, fill holes, and fix normal orientation.

    Args:
        mesh: Raw mesh from Marching Cubes (may be None upstream).
        keep_largest: Discard all but the largest connected component.
        fill_holes: Attempt to close small holes.

    Returns:
        Cleaned mesh, or None if the input was None/empty.
    """
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None

    # Модерен trimesh синтаксис за изчистване на геометрията
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    if keep_largest:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda c: len(c.faces))

    if fill_holes:
        mesh.fill_holes()

    mesh.fix_normals()

    print(f"make_printable: V={len(mesh.vertices)} F={len(mesh.faces)} "
          f"watertight={mesh.is_watertight} volume={mesh.volume:.4f}")
    return mesh


if __name__ == "__main__":
    # Two disjoint boxes -> keep_largest should return a single box.
    big = trimesh.creation.box(extents=(2, 2, 2))
    small = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    small.apply_translation([5, 0, 0])
    combined = trimesh.util.concatenate([big, small])
    cleaned = make_printable(combined)
    print("postprocess.py self-test components after clean:",
          len(cleaned.split(only_watertight=False)))
    assert len(cleaned.split(only_watertight=False)) == 1
    assert cleaned.is_watertight