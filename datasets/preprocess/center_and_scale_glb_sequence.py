#!/usr/bin/env python3
"""Center and scale sequences of GLB frames representing animated motion.

Given an input directory with subdirectories that contain GLB files named like
`frame_*.glb`, this script computes a global bounding box per sequence and
applies a shared translation + scale to every frame so the motion is centered
and fits within a fixed max-extent box (size 2.0). The transformed GLB files are
written to a mirrored directory structure under the specified output directory.

python datasets/preprocess/center_and_scale_glb_sequence.py \
    anime_test \
    ./anime_test_centered_scaled \
    --frame-limit 32 --num-samples 400 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import trimesh
from tqdm.auto import tqdm


FrameScene = Tuple[Path, trimesh.Scene]
TARGET_MAX_EXTENT = 1.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Center and scale sequences of GLB frames using a shared motion bbox."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory whose subdirectories contain frame_*.glb files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where centered frames will be written (mirrors input structure).",
    )
    parser.add_argument(
        "--pattern",
        default="frame_*.glb",
        help="Glob pattern for frame files inside each sequence directory (default: frame_*.glb).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the output directory.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="If provided, randomly center only this many motion subdirectories.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=64,
        help="Process at most this many frames per sequence (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for selecting motion subdirectories.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def find_sequence_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")
    return sorted([p for p in root.iterdir() if p.is_dir()])


def load_frame(path: Path) -> trimesh.Scene:
    scene = trimesh.load(path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    return scene


def compute_global_bounds(frames: Iterable[FrameScene]) -> Tuple[np.ndarray, np.ndarray]:
    min_corner = None
    max_corner = None

    for _, scene in frames:
        bounds = np.asarray(scene.bounds, dtype=np.float64)
        if not np.isfinite(bounds).all():
            logging.debug("Skipping bounds update due to non-finite values.")
            continue

        if min_corner is None:
            min_corner = bounds[0]
            max_corner = bounds[1]
        else:
            min_corner = np.minimum(min_corner, bounds[0])
            max_corner = np.maximum(max_corner, bounds[1])

    if min_corner is None or max_corner is None:
        logging.debug("No valid geometry found; using unit bounds around origin.")
        return np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)

    return min_corner, max_corner


def center_and_scale_sequence(
    sequence_dir: Path,
    output_dir: Path,
    pattern: str,
    overwrite: bool,
    frame_limit: int | None,
) -> None:
    target_dir = output_dir / sequence_dir.name

    if target_dir.exists() and not overwrite:
        logging.info(
            "Skipping %s as output directory exists. Use --overwrite to reprocess.",
            sequence_dir.name,
        )
        return

    frames_paths = sorted(sequence_dir.glob(pattern))
    if frame_limit is not None:
        frames_paths = frames_paths[:frame_limit]
    if not frames_paths:
        logging.warning("No frames found in %s with pattern '%s'.", sequence_dir, pattern)
        return

    scenes: List[FrameScene] = []
    for frame_path in frames_paths:
        try:
            scene = load_frame(frame_path)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to load %s: %s", frame_path, exc)
            continue
        scenes.append((frame_path, scene))

    if not scenes:
        logging.warning("No valid frames loaded for %s.", sequence_dir)
        return

    min_corner, max_corner = compute_global_bounds(scenes)
    center = (min_corner + max_corner) / 2.0
    extents = max_corner - min_corner
    max_extent = float(np.max(extents)) if np.isfinite(extents).all() else 0.0
    if max_extent <= 0.0:
        scale_factor = 1.0
    else:
        scale_factor = float(TARGET_MAX_EXTENT) / max_extent

    translation = -center
    logging.info(
        "Centering %s with translation %s and scale %.6f",
        sequence_dir.name,
        translation.round(6),
        scale_factor,
    )

    target_dir.mkdir(parents=True, exist_ok=True)

    for frame_path, scene in scenes:
        transformed = scene.copy()
        transformed.apply_translation(translation)
        transformed.apply_scale(scale_factor)

        destination = target_dir / frame_path.name
        if destination.exists() and not overwrite:
            logging.error(
                "Destination %s exists. Use --overwrite to replace existing files.", destination
            )
            continue

        try:
            transformed.export(destination, file_type="glb")
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to export %s: %s", destination, exc)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if args.output_dir.resolve() == args.input_dir.resolve():
        raise ValueError("Output directory must be different from input directory.")

    sequences = find_sequence_dirs(args.input_dir)
    if not sequences:
        logging.warning("No subdirectories found in %s.", args.input_dir)
        return

    if args.num_samples is not None:
        if args.num_samples < 1:
            raise ValueError("--num-samples must be strictly positive.")
        rng = np.random.default_rng(args.seed)
        if args.num_samples >= len(sequences):
            logging.info(
                "Requested %d samples but only %d sequences available; processing all.",
                args.num_samples,
                len(sequences),
            )
        else:
            indices = rng.choice(len(sequences), size=args.num_samples, replace=False)
            sequences = [sequences[i] for i in np.asarray(indices, dtype=int)]

    frame_limit = args.frame_limit if args.frame_limit and args.frame_limit > 0 else None

    for sequence_dir in tqdm(sequences, desc="Centering/scaling sequences"):
        center_and_scale_sequence(
            sequence_dir,
            args.output_dir,
            args.pattern,
            args.overwrite,
            frame_limit,
        )


if __name__ == "__main__":
    main()
