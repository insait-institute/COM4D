#!/usr/bin/env python3
"""Center + scale standalone GLBs under a folder and write to <name>/mesh.glb.

Example:
python datasets/preprocess/center_and_scale_glb_folder.py \
    --input /path/to/glb_root \
    --output /path/to/normalized_glbs \
    --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import trimesh

# Limit math libraries' threads per worker to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

from src.utils.data_utils import normalize_mesh


def _collect_glb_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    glb_files: List[str] = []
    for root, _, files in os.walk(input_path):
        for name in files:
            if name.lower().endswith(".glb"):
                glb_files.append(os.path.join(root, name))
    return sorted(glb_files)


def _output_path(output_root: str, glb_path: str, input_root: str) -> Tuple[str, str]:
    rel_path = os.path.relpath(glb_path, input_root)
    rel_no_ext = os.path.splitext(rel_path)[0]
    parts = [p for p in rel_no_ext.replace("\\", "/").split("/") if p]
    name = "_".join(parts) if parts else os.path.splitext(os.path.basename(glb_path))[0]
    out_dir = os.path.join(output_root, name)
    out_file = os.path.join(out_dir, "mesh.glb")
    return out_dir, out_file


def _process_one(args: Tuple[str, str, bool, str]) -> Tuple[bool, str, str]:
    glb_path, output_root, overwrite, input_root = args
    try:
        out_dir, out_file = _output_path(output_root, glb_path, input_root)
        if not overwrite and os.path.exists(out_file):
            return True, glb_path, ""
        os.makedirs(out_dir, exist_ok=True)
        mesh = trimesh.load(glb_path, force="scene")
        mesh = normalize_mesh(mesh)
        mesh.export(out_file, file_type="glb")
        return True, glb_path, ""
    except Exception as exc:  # noqa: BLE001
        return False, glb_path, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize GLBs in a folder and write each to <name>/mesh.glb."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to a .glb or a directory.")
    parser.add_argument("--output", type=str, required=True, help="Output root directory.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mesh.glb files.")
    args = parser.parse_args()

    glb_files = _collect_glb_files(args.input)
    if not glb_files:
        print(f"No .glb files found under: {args.input}")
        sys.exit(1)

    input_root = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    os.makedirs(args.output, exist_ok=True)

    worker_count = max(1, int(args.workers))
    task_args = [(path, args.output, args.overwrite, input_root) for path in glb_files]

    if worker_count == 1:
        for path in tqdm(glb_files, desc="Normalizing GLBs"):
            ok, _, err = _process_one((path, args.output, args.overwrite, input_root))
            if not ok:
                print(f"[WARN] Failed {path}: {err}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as ex:
            for ok, path, err in tqdm(
                ex.map(_process_one, task_args),
                total=len(task_args),
                desc="Normalizing GLBs",
            ):
                if not ok:
                    print(f"[WARN] Failed {path}: {err}")


if __name__ == "__main__":
    main()
