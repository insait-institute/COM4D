"""
python infer_utils/mesh_to_point.py \
    --input /work/berke_gokmen/data-1/animesh/humanoids_scaled/ \
    --output /work/berke_gokmen/data-1/animesh/humanoids_scaled_preprocessed/ \
    --workers 4 \

python infer_utils/mesh_to_point.py \
    --input /work/berke_gokmen/data-1/animesh/animals_scaled/ \
    --output /work/berke_gokmen/data-1/animesh/animals_scaled_preprocessed/ \
    --workers 4 \
"""


import os
import trimesh
import numpy as np
import argparse
import json
import sys
import concurrent.futures
from functools import partial
import shutil
from pathlib import Path

# Limit math libraries' threads per worker to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, **kwargs):
        # Minimal fallback when tqdm is unavailable
        return iterable if iterable is not None else range(total or 0)

from src.utils.data_utils import scene_to_parts, mesh_to_surface, normalize_mesh


def is_valid_glb(path: str) -> bool:
    """Return True if path is a top-level .glb file we should process.
    Rules: must be a file, extension .glb, and base name must not end with `_full`.
    """
    if not os.path.isfile(path):
        return False
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    if ext.lower() != ".glb":
        return False
    if name.endswith("_full"):
        return False
    return True


def try_reuse_preprocessed(glb_path: str, output_root: str, out_dir_name: str, reuse_dir: str) -> bool:
    """If a legacy preprocessed folder exists under `reuse_dir/<room_name>` and its
    num_parts.json has a `mesh_path` that matches `glb_path`, copy the cached files
    into `<output_root>/<out_dir_name>` and return True. Otherwise return False.
    Legacy layout uses only room name folders (no UUID prefix).
    """
    if not reuse_dir:
        return False
    room_name = os.path.splitext(os.path.basename(glb_path))[0]
    legacy_dir = os.path.join(reuse_dir, room_name)
    legacy_json = os.path.join(legacy_dir, "num_parts.json")
    legacy_points = os.path.join(legacy_dir, "points.npy")
    if not os.path.isdir(legacy_dir):
        return False
    if not (os.path.isfile(legacy_json) and os.path.isfile(legacy_points)):
        return False
    try:
        with open(legacy_json, "r") as f:
            meta = json.load(f)
        legacy_mesh = os.path.abspath(meta.get("mesh_path", ""))
        cur_mesh = os.path.abspath(glb_path)
        if legacy_mesh != cur_mesh:
            return False
    except Exception:
        return False
    # Paths match: copy to new output folder
    out_dir = os.path.join(output_root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(legacy_points, os.path.join(out_dir, "points.npy"))
    shutil.copy2(legacy_json, os.path.join(out_dir, "num_parts.json"))
    return True


def output_already_processed(output_root: str, folder_name: str) -> bool:
    out_dir = os.path.join(output_root, folder_name)
    points_path = os.path.join(out_dir, "points.npy")
    meta_path = os.path.join(out_dir, "num_parts.json")
    return os.path.isfile(points_path) and os.path.isfile(meta_path)


def _process_task(args):
    """Wrapper for parallel execution. Returns (ok, path, out_dir_name, error_msg, action)."""
    fpath, output_root, out_name, reuse_dir, normalize = args
    folder_name = out_name if out_name else os.path.splitext(os.path.basename(fpath))[0]
    try:
        if output_already_processed(output_root, folder_name):
            return True, fpath, folder_name, None, "skipped"
        if reuse_dir:
            reused = try_reuse_preprocessed(fpath, output_root, out_name, reuse_dir)
            if reused:
                return True, fpath, out_name, None, "reused"
        processed = process_one(
            fpath,
            output_root,
            out_dir_name=out_name,
            normalize=bool(normalize),
        )
        action = "processed" if processed else "skipped"
        return True, fpath, folder_name, None, action
    except Exception as e:
        return False, fpath, out_name, str(e), "error"


def process_one(
    input_path: str,
    output_root: str,
    out_dir_name: str | None = None,
    normalize: bool = False,
) -> bool:
    """Process a single GLB mesh and write outputs into
    `<output_root>/<out_dir_name or mesh_name>/points.npy` and `num_parts.json`.
    The JSON includes `num_parts` and `mesh_path` (absolute path).
    """
    assert os.path.exists(input_path), f"{input_path} does not exist"

    mesh_name = os.path.basename(input_path).split(".")[0]
    folder_name = out_dir_name if out_dir_name else mesh_name
    if output_already_processed(output_root, folder_name):
        return False
    out_dir = os.path.join(output_root, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "num_parts": 0,
        "mesh_path": os.path.abspath(input_path),
    }

    # sample points from mesh surface
    with open(input_path, "rb") as f:
        mesh = trimesh.load(file_obj=f, file_type="glb", process=False)
    if normalize:
        mesh = normalize_mesh(mesh)

    # If it's a Scene it has `.geometry`; if it's a single Trimesh, use 1 as a fallback
    num_parts = len(mesh.geometry) if hasattr(mesh, "geometry") else 1
    config["num_parts"] = num_parts

    if num_parts > 1 and num_parts <= 16 and hasattr(mesh, "geometry"):
        parts = scene_to_parts(
            mesh,
            return_type="point",
            normalize=False,
        )
        scene_like = mesh
    else:
        parts = []
        scene_like = mesh

    # Convert to geometry for surface sampling
    mesh_geom = scene_like.to_geometry() if hasattr(scene_like, "to_geometry") else scene_like
    obj = mesh_to_surface(mesh_geom, return_dict=True)

    datas = {
        "object": obj,
        "parts": parts,
    }

    # save points
    np.save(os.path.join(out_dir, "points.npy"), datas)

    # save config with mesh_path included
    with open(os.path.join(out_dir, "num_parts.json"), "w") as f:
        json.dump(config, f, separators=(",", ":"))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="assets/objects/scissors.glb",
        help="Path to a GLB file or a directory structured as <root>/<uuid>/{room_name}.glb (one level).",
    )
    parser.add_argument("--output", type=str, default="preprocessed_data")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, (os.cpu_count() or 2)),
        help="Number of parallel worker processes (default: min(8, CPU count)). Use 1 for no parallelism.",
    )
    parser.add_argument(
        "--reuse-dir",
        type=str,
        default=None,
        help="Path to a legacy preprocessed directory to reuse existing data. ",
    )
    parser.add_argument(
        "--normalize",
        type=int,
        default=0,
        help="Whether to normalize the mesh during processing.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    args.workers = min(args.workers, os.cpu_count())

    print("Using workers:", args.workers, "cpu count", os.cpu_count())

    # Create the root output directory once
    os.makedirs(output_path, exist_ok=True)

    if os.path.isdir(input_path):
        # Search exactly one level: <input>/<uuid>/{room_name}.glb
        # Output folder name becomes {uuid}_{room_name} to avoid collisions across identical room names.
        tasks = []  # list of tuples (glb_path, out_dir_name)

        for uuid_entry in sorted(os.listdir(input_path)):
            uuid_path = os.path.join(input_path, uuid_entry)
            if not os.path.isdir(uuid_path):
                continue
            # Look for files directly under <uuid>: {room_name}.glb (exclude names ending with _full)
            for f in sorted(os.listdir(uuid_path)):
                fpath = os.path.join(uuid_path, f)
                if is_valid_glb(fpath):
                    room_name = os.path.splitext(os.path.basename(fpath))[0]
                    tasks.append((fpath, f"{uuid_entry}_{room_name}"))

        if not tasks:
            print(
                f"No valid {input_path}/<uuid>/*.glb files found. Files ending with '_full' are ignored."
            )

        print(f"Found {len(tasks)} GLB files to process.", tasks[:5], "..." if len(tasks) > 5 else "")

        # Execute tasks
        if not tasks:
            pass
        else:
            worker_count = max(1, int(args.workers))
            if worker_count == 1:
                # Sequential with progress bar
                for fpath, out_name in tqdm(tasks, total=len(tasks), desc="Processing GLBs"):
                    ok, _, folder_name, err, action = _process_task(
                        (fpath, output_path, out_name, args.reuse_dir, args.normalize)
                    )
                    if not ok:
                        print(f"[WARN] Skipping {fpath} due to error: {err}")
                    elif action == "reused":
                        print(f"[REUSE] {fpath} -> {out_name}")
                    elif action == "skipped":
                        print(f"[SKIP] {fpath} -> {folder_name} already processed.")
            else:
                # Parallel with processes using map (lower overhead) and chunking
                task_args = [(f, output_path, n, args.reuse_dir, args.normalize) for (f, n) in tasks]
                total = len(task_args)
                with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as ex:
                    for ok, fpath, out_name, err, action in tqdm(
                        ex.map(_process_task, task_args),
                        total=total,
                        desc="Processing GLBs",
                    ):
                        if not ok:
                            print(f"[WARN] Skipping {fpath} -> {out_name} due to error: {err}")
                        elif action == "reused":
                            print(f"[REUSE] {fpath} -> {out_name}")
                        elif action == "skipped":
                            print(f"[SKIP] {fpath} -> {out_name} already processed.")
    else:
        if not is_valid_glb(input_path):
            raise ValueError(
                f"Input must be a .glb file (not ending with _full) or a directory. Got: {input_path}"
            )
        processed = process_one(
            input_path,
            output_path,
            out_dir_name=None,
            normalize=bool(args.normalize),
        )
        if not processed:
            print(f"[SKIP] {input_path} already processed.")
