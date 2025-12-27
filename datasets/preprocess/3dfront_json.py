#!/usr/bin/env python3
"""
Build a JSON manifest for preprocessed 3D-FRONT rooms.

Input assumptions:
  - --input points to the preprocessed root that contains folders like:
        <uuid>_<room_name>/
            points.npy (or points.npz)
            num_parts.json (optional)
  - --render-root points to the render root that contains folders like:
        <uuid>/<room_name>/
            rerender_0000.webp ... rerender_0010.webp
  - --glb-root is optional; if provided we will try to resolve GLB paths from
    <glb_root>/<uuid>/<room_name>.glb or <glb_root>/<folder_name>.glb

Output:
  - Writes a JSON array where each element contains:
        {
          "file": "<mesh_name>.glb" or folder name,
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


def _resolve_mesh_path(glb_root: Optional[Path], folder_name: str) -> Optional[Path]:
    if glb_root is None:
        return None
    if "_" in folder_name:
        uuid, room = folder_name.split("_", 1)
        candidate = glb_root / uuid / f"{room}.glb"
        if candidate.exists():
            return candidate.resolve()
    candidate = glb_root / f"{folder_name}.glb"
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
    for folder_name in tqdm(sorted(os.listdir(preprocessed_root)), desc="Rooms"):
        folder_path = preprocessed_root / folder_name
        if not folder_path.is_dir():
            continue

        surface_path = folder_path / "points.npy"
        if not surface_path.exists():
            surface_path = folder_path / "points.npz"
        if not surface_path.exists():
            continue

        num_parts = _load_num_parts(folder_path / "num_parts.json")
        mesh_path = _resolve_mesh_path(glb_root, folder_name)

        render_dir = render_root / folder_name
        if "_" in folder_name:
            uuid, room = folder_name.split("_", 1)
            render_dir = render_root / uuid / room

        for idx in range(0, 11):
            image_path = render_dir / f"rerender_{idx:04d}.webp"
            if not image_path.exists():
                continue
            image_path = image_path.resolve()

            file_name = f"{folder_name}.glb" if mesh_path is not None else folder_name
            entries.append(
                {
                    "file": file_name,
                    "num_parts": num_parts,
                    "valid": True,
                    "mesh_path": str(mesh_path) if mesh_path is not None else None,
                    "surface_path": str(surface_path.resolve()),
                    "image_path": str(image_path),
                    "iou_mean": 0.0,
                    "iou_max": 0.0,
                }
            )
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a JSON manifest for preprocessed 3D-FRONT rooms.")
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root directory containing preprocessed folders with points.npy/num_parts.json.",
    )
    ap.add_argument(
        "--render-root",
        type=Path,
        required=True,
        help="Root directory containing per-room rerender_0000.webp images.",
    )
    ap.add_argument(
        "--glb-root",
        type=Path,
        default=None,
        help="Optional root directory containing original GLB files.",
    )
    ap.add_argument("--output", type=Path, default=Path("3dfront_manifest.json"), help="Output JSON path.")
    args = ap.parse_args()

    entries = build_manifest(args.input, args.render_root, args.glb_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(entries, f, indent=2)

    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
