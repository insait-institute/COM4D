#!/usr/bin/env python3
"""
Build a dataset index JSON of the form:

{
    "abe_coverToStand": [
        {
            "surface_path": "surface_path",
            "image_path": "image_path",
            "iou_mean": 0.0,
            "iou_max": 0.0
        },
        ...
    ],
    ...
}


python datasets/preprocess/deformingthings_json.py \
        --pair /data/animesh/humanoids_scaled_preprocessed:/data/animesh/humanoids_scaled_render \
        --pair /data/animesh/animals_scaled_preprocessed:/data/animesh/animals_scaled_render \
        -o dataset_local/animesh_scaled_local.json --pretty

Usage examples:
    python datasets/preprocess/deformingthings_json.py \
        --pair /data/animated-mesh-part-preprocessed/humanoids:/data/animated-mesh-render/humanoids \
        -o dataset_local

    # Multiple input directory pairs:
    python datasets/preprocess/deformingthings_json.py \
        --pair /data/animated-mesh-preprocessed-not-centered/humanoids:/data/animated-mesh-render-not-centered/humanoids \
        --pair /data/animated-mesh-preprocessed-not-centered/animals:/data/animated-mesh-render-not-centered/animals \
        -o dataset_local/animated_mesh_not_centered_local_last.json --pretty
    
    python datasets/preprocess/deformingthings_json.py \
        --pair mini_animated_dataset/animated-mesh-preprocessed-not-centered:mini_animated_dataset/animated-mesh-render-original-no-bg \
        -o dataset_local/mini_animated_mesh_not_centered_original_local.json --pretty
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

# Accept optional trailing "_rendering"
SUBDIR_PATTERN_PREPROC = re.compile(r"^(?P<base>.+)_frame_(?P<frame>\d+)(?:_rendering)?$")
SUBDIR_PATTERN_FRAME = re.compile(r"^(?P<base>.+)_frame_(?P<frame>\d+)(?:_rendering)?$")
RENDER_FILE_PATTERN = re.compile(r"^frame_(?P<frame>\d+)\.(?P<ext>png|jpg|jpeg)$", re.IGNORECASE)

def parse_base_frame(dirname: str):
    m = SUBDIR_PATTERN_FRAME.match(dirname)
    if not m:
        return None
    return m.group("base"), m.group("frame")

def parse_base_preproc(dirname: str):
    m = SUBDIR_PATTERN_PREPROC.match(dirname)
    if not m:
        return None
    return m.group("base"), m.group("frame")

def collect_preproc(preproc_root: Path, verbose: bool = True) -> Dict[Tuple[str, str], str]:
    """
    Find all (base, frame) -> surface_path within preproc_root.
    Looks for 'points.npy' and reads its parent directory name.
    """
    results: Dict[Tuple[str, str], str] = {}
    if not preproc_root.exists():
        if verbose:
            print(f"[WARN] Preproc root does not exist: {preproc_root}", file=sys.stderr)
        return results

    # Recursively find points.npy
    for points in preproc_root.rglob("points.npy"):
        parent = points.parent
        bf = parse_base_preproc(parent.name)
        if not bf:
            continue
        key = (bf[0], bf[1].zfill(4))  # (base, frame)
        # Prefer shortest path deterministically if duplicates
        prev = results.get(key)
        if prev is None or len(str(points)) < len(prev):
            results[key] = str(points.resolve())
    return results

def collect_render(render_root: Path, verbose: bool = True) -> Dict[Tuple[str, str], str]:
    """
    Find all (base, frame) -> image_path within render_root.
    Supports legacy `<name>_frame_xxxx/rendering.png` and new `<name>/frame_xxxx.png` layouts.
    """
    results: Dict[Tuple[str, str], str] = {}
    if not render_root.exists():
        if verbose:
            print(f"[WARN] Render root does not exist: {render_root}", file=sys.stderr)
        return results

    for img in render_root.rglob("*"):
        if not img.is_file():
            continue
        name_lower = img.name.lower()
        match_key = None

        if name_lower == "rendering.png":
            parent = img.parent
            bf = parse_base_frame(parent.name)
            if not bf:
                continue
            match_key = (bf[0], bf[1].zfill(4))
        else:
            fm = RENDER_FILE_PATTERN.match(img.name)
            if not fm:
                continue
            base = img.parent.name
            frame = fm.group("frame").zfill(4)
            match_key = (base, frame)

        prev = results.get(match_key)
        if prev is None or len(str(img)) < len(prev):
            results[match_key] = str(img.resolve())
    return results

def build_index(pairs, strict=False, verbose=True):
    """
    Build the index from a list of (preproc_root, render_root) pairs.

    strict=False: skip missing counterparts with a warning.
    strict=True:  raise on any missing expected file.
    """
    grouped = defaultdict(list)

    for preproc_root, render_root in pairs:
        preproc_root = Path(preproc_root).resolve()
        render_root = Path(render_root).resolve()
        if verbose:
            print(f"[INFO] Scanning pair:\n  preproc: {preproc_root}\n  render : {render_root}")

        preproc_map = collect_preproc(preproc_root, verbose=verbose)
        render_map = collect_render(render_root, verbose=verbose)

        if verbose:
            # print(list(preproc_map.keys())[:10], list(render_map.keys())[:10])
            print(f"[INFO] Found in preproc: {len(preproc_map)} frames, render: {len(render_map)} frames")

        # Intersect keys so we only include frames that have both files
        # Normalize function: (name, index_str) -> (name, int(index_str))
        def normalize_keys(d):
            return {(name, int(idx)) for (name, idx) in d.keys()}

        common_keys = normalize_keys(preproc_map) & normalize_keys(render_map)

        if strict:
            # If strict, ensure there are no preproc-only or render-only frames
            only_preproc = set(preproc_map.keys()) - set(render_map.keys())
            only_render = set(render_map.keys()) - set(preproc_map.keys())
            if only_preproc:
                missing = "\n  ".join([f"{b}_frame_{f}" for b, f in sorted(only_preproc)])
                raise FileNotFoundError(f"Missing renderings for frames:\n  {missing}")
            if only_render:
                missing = "\n  ".join([f"{b}_frame_{f}" for b, f in sorted(only_render)])
                raise FileNotFoundError(f"Missing preprocess (points.npy) for frames:\n  {missing}")
        else:
            if verbose:
                if len(common_keys) == 0 and (len(preproc_map) > 0 or len(render_map) > 0):
                    print("[WARN] No intersecting frames with both points.npy and rendered images", file=sys.stderr)

        for (base, frame) in sorted(common_keys, key=lambda x: (x[0], int(x[1]))):
            grouped[base].append({
                "surface_path": preproc_map[(base, str(frame).zfill(4))],
                "image_path": render_map[(base, str(frame).zfill(4))],
                "iou_mean": 0.0,
                "iou_max": 0.0,
            })

    # Ensure frames are sorted numerically per sequence
    for base in list(grouped.keys()):
        def frame_key(item):
            # Robustly extract numeric frame back from path's parent name
            parent_name = Path(item["surface_path"]).parent.name
            bf = parse_base_frame(parent_name)
            if bf:
                return int(bf[1])
            return 0
        grouped[base].sort(key=frame_key)

    return grouped

def parse_pairs(pair_args):
    pairs = []
    for arg in pair_args:
        if ":" not in arg:
            raise ValueError(f"--pair must be in PREPROC:RENDER form, got: {arg}")
        preproc, render = arg.split(":", 1)
        pairs.append((preproc, render))
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Build animated mesh JSON index (robust, recursive).")
    parser.add_argument(
        "--pair", action="append", required=True,
        help="Input pair PREPROC_ROOT:RENDER_ROOT (repeatable)."
    )
    parser.add_argument("-o", "--output", required=True, help="Output JSON file path.")
    parser.add_argument("--strict", action="store_true", help="Error on missing counterparts.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    args = parser.parse_args()

    try:
        pairs = parse_pairs(args.pair)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    index = build_index(pairs, strict=args.strict, verbose=not args.quiet)

    # Deterministic key order
    ordered = {k: index[k] for k in sorted(index.keys())}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(ordered, f, indent=4, ensure_ascii=False)
        else:
            json.dump(ordered, f, separators=(",", ":"), ensure_ascii=False)

    if not args.quiet:
        total_frames = sum(len(v) for v in ordered.values())
        print(f"[INFO] Wrote {out_path} with {len(ordered)} sequences and {total_frames} frames.")

if __name__ == "__main__":
    main()
