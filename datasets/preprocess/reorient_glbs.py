"""
python datasets/preprocess/reorient_glbs.py \
    --input ./anime_test \
    --workers 8
"""

import os
import argparse
import concurrent.futures
import math
import shutil
from typing import Tuple, Dict, List, Optional

import numpy as np
import trimesh

# Keep math libs from oversubscribing when multiprocessing
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


def is_valid_glb(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    name, ext = os.path.splitext(os.path.basename(path))
    folder_name = os.path.basename(os.path.dirname(path))
    if ext.lower() != ".glb":
        return False
    # Keep parity with other preprocessing scripts: ignore *_full.glb
    if name.endswith("_full"):
        return False
    return True


def _axis_vector(axis: str) -> Tuple[float, float, float]:
    a = axis.lower()
    if a == "x":
        return (1.0, 0.0, 0.0)
    if a == "y":
        return (0.0, 1.0, 0.0)
    if a == "z":
        return (0.0, 0.0, 1.0)
    raise ValueError(f"Invalid axis '{axis}'. Choose from x,y,z.")


def _scene_bounds(s: trimesh.Scene | trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(s, "bounds"):
        b = s.bounds
        # bounds shape (2,3) min,max
        if isinstance(b, np.ndarray) and b.shape == (2, 3):
            return b[0].astype(float), b[1].astype(float)
    # Fallbacks
    c = getattr(s, "centroid", None)
    if isinstance(c, (np.ndarray, list, tuple)) and len(c) == 3:
        c = np.asarray(c, dtype=float)
        return c.copy(), c.copy()
    z = np.zeros(3, dtype=float)
    return z.copy(), z.copy()


def _scene_center(s: trimesh.Scene | trimesh.Trimesh) -> np.ndarray:
    mins, maxs = _scene_bounds(s)
    return (mins + maxs) / 2.0


def reorient_one(
    path: str,
    angle_deg: float,
    axis: str,
    backup: bool,
    pivot: Optional[np.ndarray] = None,
) -> None:
    # Load scene/mesh
    with open(path, "rb") as f:
        scene_or_mesh = trimesh.load(file_obj=f, file_type="glb", process=False)

    # Calculate rotation around the chosen pivot (default: per-file center)
    center = np.asarray(pivot, dtype=float) if pivot is not None else _scene_center(scene_or_mesh)
    direction = np.asarray(_axis_vector(axis), dtype=float)
    angle_rad = math.radians(angle_deg)
    T = trimesh.transformations.rotation_matrix(angle_rad, direction, point=center)

    # Apply in-place
    if hasattr(scene_or_mesh, "apply_transform"):
        scene_or_mesh.apply_transform(T)
    else:
        # Convert to geometry as fallback
        try:
            geom = scene_or_mesh.to_geometry()  # type: ignore[attr-defined]
            geom.apply_transform(T)
            scene_or_mesh = geom
        except Exception:
            raise RuntimeError("Object does not support transform application")

    # Write back atomically
    tmp_path = path + ".tmp.glb"
    if backup:
        backup_path = path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(path, backup_path)
    scene_or_mesh.export(tmp_path, file_type="glb")
    os.replace(tmp_path, path)


def _process_task(args_tuple):
    path, angle_deg, axis, backup, pivot = args_tuple
    try:
        reorient_one(path, angle_deg, axis, backup, pivot)
        return True, path, None
    except Exception as e:
        return False, path, str(e)


def collect_glbs(input_path: str) -> list[str]:
    files: list[str] = []
    print(f"Collecting GLB files from: {input_path}")
    if os.path.isdir(input_path):
        # one-level deep search: <input>/<uuid>/*.glb to match existing pattern
        for uuid in sorted(os.listdir(input_path)):
            upath = os.path.join(input_path, uuid)
            if not os.path.isdir(upath):
                continue
            for f in sorted(os.listdir(upath)):
                fpath = os.path.join(upath, f)
                if is_valid_glb(fpath):
                    files.append(fpath)
        # Also include top-level *.glb if present (useful for simple folders)
        for f in sorted(os.listdir(input_path)):
            fpath = os.path.join(input_path, f)
            if is_valid_glb(fpath) and fpath not in files:
                files.append(fpath)
    else:
        if not is_valid_glb(input_path):
            raise ValueError(f"Input must be a .glb or directory. Got: {input_path}")
        files.append(input_path)
    return files


def group_by_parent(files: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for f in files:
        d = os.path.dirname(f)
        groups.setdefault(d, []).append(f)
    # Ensure deterministic order within groups
    for d in groups:
        groups[d].sort()
    return groups


def compute_group_pivot(files: List[str], mode: str) -> np.ndarray:
    """Compute a consistent rotation pivot for a group of GLBs.

    - mode == 'origin': use (0,0,0)
    - mode == 'group': use union bbox center over files
    - mode == 'per-file': unused here (handled at call-site)
    """
    if mode == "origin":
        return np.zeros(3, dtype=float)
    # Union bounds across files
    global_min = np.array([np.inf, np.inf, np.inf], dtype=float)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for path in files:
        try:
            with open(path, "rb") as f:
                som = trimesh.load(file_obj=f, file_type="glb", process=False)
            mins, maxs = _scene_bounds(som)
            global_min = np.minimum(global_min, mins)
            global_max = np.maximum(global_max, maxs)
        except Exception:
            continue
    if not np.isfinite(global_min).all() or not np.isfinite(global_max).all():
        return np.zeros(3, dtype=float)
    return (global_min + global_max) / 2.0


def main():
    parser = argparse.ArgumentParser(description="Reorient GLB files by a fixed rotation, in-place.")
    parser.add_argument("--input", type=str, required=True, help="Path to a .glb or a directory")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, (os.cpu_count() or 2)),
        help="Parallel worker processes (default: min(8, CPU count))",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="x",
        choices=["x", "y", "z"],
        help="Axis to rotate around (default: x)",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=-90.0,
        help="Angle in degrees (default: -90). Use +90/-90 to flip from top-view to front-view.",
    )
    parser.add_argument(
        "--pivot_mode",
        type=str,
        default="group",
        choices=["group", "per-file", "origin"],
        help=(
            "Pivot for rotation: 'group' uses union bbox center per directory (default), "
            "'per-file' uses each file's own center (old behavior), 'origin' uses (0,0,0)."
        ),
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Keep a .bak copy next to each modified file",
    )
    args = parser.parse_args()

    args.workers = max(1, min(args.workers, (os.cpu_count() or 1)))

    files = collect_glbs(args.input)
    if not files:
        print("No GLB files found to reorient.")
        return

    print(
        f"Found {len(files)} GLB files. Rotating {args.angle}° around {args.axis}-axis (pivot_mode={args.pivot_mode})."
    )

    if args.pivot_mode == "per-file":
        # Original behavior: each file uses its own center
        if args.workers == 1:
            for f in tqdm(files, total=len(files), desc="Reorienting GLBs"):
                ok, path, err = _process_task((f, args.angle, args.axis, args.backup, None))
                if not ok:
                    print(f"[WARN] Failed {path}: {err}")
        else:
            task_args = [(f, args.angle, args.axis, args.backup, None) for f in files]
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                for ok, path, err in tqdm(
                    ex.map(_process_task, task_args), total=len(task_args), desc="Reorienting GLBs"
                ):
                    if not ok:
                        print(f"[WARN] Failed {path}: {err}")
        return

    # Group-aware pivot: compute pivot per parent directory
    groups = group_by_parent(files)
    tasks = []
    group_items = list(groups.items())
    if group_items:
        print(f"Computing group pivots for {len(group_items)} groups...")
    for parent, gfiles in tqdm(group_items, total=len(group_items), desc="Computing pivots"):
        pivot = compute_group_pivot(gfiles, args.pivot_mode)
        for f in gfiles:
            tasks.append((f, args.angle, args.axis, args.backup, pivot))

    if args.workers == 1:
        for args_tuple in tqdm(tasks, total=len(tasks), desc="Reorienting GLBs"):
            ok, path, err = _process_task(args_tuple)
            if not ok:
                print(f"[WARN] Failed {path}: {err}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for ok, path, err in tqdm(
                ex.map(_process_task, tasks), total=len(tasks), desc="Reorienting GLBs"
            ):
                if not ok:
                    print(f"[WARN] Failed {path}: {err}")


if __name__ == "__main__":
    main()
