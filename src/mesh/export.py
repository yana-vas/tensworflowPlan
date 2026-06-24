"""Export trimesh meshes to STL/OBJ/PLY. STL is the 3D-printing target."""

from pathlib import Path
from typing import Optional

import trimesh


class MeshExporter:

    def __init__(self, create_dirs: bool = True) -> None:
        self.create_dirs = create_dirs

    def _prepare_path(self, path: str) -> Path:
        p = Path(path)
        if self.create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def export_stl(self, mesh: trimesh.Trimesh, path: str, binary: bool = True) -> bool:
        try:
            out = self._prepare_path(path)
            mesh.export(str(out), file_type="stl" if binary else "stl_ascii")
            print(f"Exported STL: {out} | V={len(mesh.vertices)} F={len(mesh.faces)} "
                  f"watertight={mesh.is_watertight}")
            return True
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"Failed to export STL: {exc}")
            return False

    def export_obj(self, mesh: trimesh.Trimesh, path: str) -> bool:
        try:
            mesh.export(str(self._prepare_path(path)), file_type="obj")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to export OBJ: {exc}")
            return False

    def export_ply(self, mesh: trimesh.Trimesh, path: str) -> bool:
        try:
            mesh.export(str(self._prepare_path(path)), file_type="ply")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to export PLY: {exc}")
            return False


def save_mesh(mesh: trimesh.Trimesh, path: str, file_format: Optional[str] = None) -> bool:
    exporter = MeshExporter()
    if file_format is None:
        ext = Path(path).suffix.lower()
        file_format = ext[1:] if ext else "stl"
    if file_format == "stl":
        return exporter.export_stl(mesh, path)
    if file_format == "obj":
        return exporter.export_obj(mesh, path)
    if file_format == "ply":
        return exporter.export_ply(mesh, path)
    print(f"Unknown format: {file_format}")
    return False


if __name__ == "__main__":
    import tempfile
    box = trimesh.creation.box(extents=(1, 1, 1))
    with tempfile.TemporaryDirectory() as d:
        ok = save_mesh(box, str(Path(d) / "box.stl"))
    print("export.py self-test:", ok)
    assert ok