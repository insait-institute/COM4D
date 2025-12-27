#!/usr/bin/env python3
"""
Convert .anime files into per-frame GLB meshes.

Example:
python datasets/preprocess/anime_to_glb.py \
    --input /work/berke_gokmen/data-1/DeformingThings4D/animals \
    --output ./anime_test \
    --workers 1
"""

import argparse
import concurrent.futures
import os
import sys
from typing import Iterable, List, Tuple

import numpy as np
import trimesh

# Limit math libraries' threads per worker to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)


def anime_read(filename: str):
    """
    filename: .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: [nv, 3], vertex data of the 1st frame
        face_data: [nt, 3], triangle face data of the 1st frame
        offset_data: [nf-1, nv, 3], 3D offset data from the 2nd to the last frame
    """
    with open(filename, "rb") as f:
        nf = np.fromfile(f, dtype=np.int32, count=1)[0]
        nv = np.fromfile(f, dtype=np.int32, count=1)[0]
        nt = np.fromfile(f, dtype=np.int32, count=1)[0]
        vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
        face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
        offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


def _collect_anime_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    anime_files: List[str] = []
    for root, _, files in os.walk(input_path):
        for name in files:
            if name.endswith(".anime"):
                anime_files.append(os.path.join(root, name))
    return sorted(anime_files)


def _output_dir(output_root: str, anime_path: str) -> str:
    parent_name = os.path.basename(os.path.dirname(anime_path))
    return os.path.join(output_root, parent_name)


def _process_one(args: Tuple[str, str, bool, int]) -> Tuple[bool, str, str]:
    anime_path, output_root, overwrite, max_frames = args
    try:
        nf, _nv, _nt, vert_data, face_data, offset_data = anime_read(anime_path)
        if max_frames > 0:
            nf = min(nf, max_frames)
        out_dir = _output_dir(output_root, anime_path)
        os.makedirs(out_dir, exist_ok=True)

        for frame_idx in range(nf):
            out_path = os.path.join(out_dir, f"frame_{frame_idx + 1:04d}.glb")
            if not overwrite and os.path.exists(out_path):
                continue
            if frame_idx == 0:
                verts = vert_data
            else:
                verts = vert_data + offset_data[frame_idx - 1]
            mesh = trimesh.Trimesh(vertices=verts, faces=face_data, process=False)
            mesh.export(out_path)
        return True, anime_path, ""
    except Exception as exc:
        return False, anime_path, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .anime files into per-frame GLBs.")
    parser.add_argument("--input", type=str, required=True, help="Path to a .anime file or a directory.")
    parser.add_argument("--output", type=str, required=True, help="Output root directory for GLB frames.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frame GLBs.")
    parser.add_argument("--max-frames", type=int, default=64, help="Max number of frames to export (0 = no limit).")
    args = parser.parse_args()

    anime_files = _collect_anime_files(args.input)
    if not anime_files:
        print(f"No .anime files found under: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    worker_count = max(1, int(args.workers))
    task_args = [(path, args.output, args.overwrite, args.max_frames) for path in anime_files]

    if worker_count == 1:
        for path in tqdm(anime_files, desc="Converting"):
            ok, _, err = _process_one((path, args.output, args.overwrite, args.max_frames))
            if not ok:
                print(f"[WARN] Failed {path}: {err}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as ex:
            for ok, path, err in tqdm(
                ex.map(_process_one, task_args),
                total=len(task_args),
                desc="Converting",
            ):
                if not ok:
                    print(f"[WARN] Failed {path}: {err}")


if __name__ == "__main__":
    main()
