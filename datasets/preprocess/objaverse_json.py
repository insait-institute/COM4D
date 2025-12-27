#!/usr/bin/env python3
"""
Build a JSON manifest for preprocessed Objaverse meshes.

Input assumptions:
  - --preprocessed-root contains folders like:
        <obj_id>_mesh/
            points.npy (or points.npz)
            num_parts.json (optional)
  - --render-root contains folders like:
        <obj_id>/
            mesh.png
  - --glb-root contains folders like:
        <obj_id>/
            mesh.glb

Output:
  - Writes a JSON array where each element contains:
        {
          "file": "<obj_id>.glb" or "<obj_id>",
          "num_parts": <int>,
          "valid": <bool>,
          "mesh_path": <abs path or None>,
          "surface_path": <abs path>,
          "image_path": <abs path>,
          "iou_mean": 0.0,
          "iou_max": 0.0
        }
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    def tqdm(iterable, **kwargs):
        return iterable


def _resolve_mesh_path(glb_root: Optional[Path], obj_id: str) -> Optional[Path]:
    if glb_root is None:
        return None
    candidate = glb_root / obj_id / "mesh.glb"
    if candidate.exists():
        return candidate.resolve()
    candidate = glb_root / f"{obj_id}.glb"
    if candidate.exists():
        return candidate.resolve()
    return None


def _load_num_parts(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return int(data.get("num_parts", 0))
    return 0


def build_manifest(
    preprocessed_root: Path,
    render_root: Path,
    glb_root: Optional[Path],
) -> List[Dict]:
    entries: List[Dict] = []
    for folder_name in tqdm(sorted(os.listdir(preprocessed_root)), desc="Objects"):
        folder_path = preprocessed_root / folder_name
        if not folder_path.is_dir():
            continue
        if folder_name.endswith("_mesh"):
            obj_id = folder_name[:-5]
        else:
            obj_id = folder_name

        surface_path = folder_path / "points.npy"
        if not surface_path.exists():
            surface_path = folder_path / "points.npz"
        if not surface_path.exists():
            continue

        num_parts = _load_num_parts(folder_path / "num_parts.json")
        mesh_path = _resolve_mesh_path(glb_root, obj_id)

        image_path = render_root / obj_id / "mesh.png"
        if not image_path.exists():
            continue

        file_name = f"{obj_id}.glb" if mesh_path is not None else obj_id
        entries.append(
            {
                "file": file_name,
                "num_parts": num_parts,
                "valid": True,
                "mesh_path": str(mesh_path) if mesh_path is not None else None,
                "surface_path": str(surface_path.resolve()),
                "image_path": str(image_path.resolve()),
                "iou_mean": 0.0,
                "iou_max": 0.0,
            }
        )
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a JSON manifest for preprocessed Objaverse meshes.")
    ap.add_argument(
        "--preprocessed-root",
        type=Path,
        required=True,
        help="Root directory containing preprocessed folders with points.npy/num_parts.json.",
    )
    ap.add_argument(
        "--render-root",
        type=Path,
        required=True,
        help="Root directory containing per-object mesh.png renders.",
    )
    ap.add_argument(
        "--glb-root",
        type=Path,
        default=None,
        help="Optional root directory containing original GLB files.",
    )
    ap.add_argument("--output", type=Path, default=Path("objaverse_manifest.json"), help="Output JSON path.")
    args = ap.parse_args()

    entries = build_manifest(args.preprocessed_root, args.render_root, args.glb_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(entries, f, indent=2)

    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
