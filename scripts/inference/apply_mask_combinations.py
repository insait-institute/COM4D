#!/usr/bin/env python3
"""
Union all masks per frame, optionally dilate or close them, and apply to the frame.

Frames live in a dedicated directory, masks in another, with masks named
`<frame_stem>_object_<idx>.png`. For each frame we merge all masks
across one or more mask directories,
apply the union as a single mask, and write the masked frame into the
output directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply every mask combination to its corresponding frame."
    )
    parser.add_argument("frames_dir", type=Path, help="Directory containing the frames.")
    parser.add_argument(
        "mask_dirs",
        type=Path,
        nargs="+",
        help="One or more directories containing the masks.",
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory where processed frames will be saved."
    )
    parser.add_argument(
        "--frame-exts",
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="Frame filename extensions to consider (case-insensitive).",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=0,
        help="Pixel values above this are treated as part of the mask.",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=0,
        help="Dilate masks by this many pixels before applying (0 means no dilation).",
    )
    parser.add_argument(
        "--closing",
        type=int,
        default=0,
        help=(
            "Apply morphological closing with this radius to fill small gaps "
            "(0 means no closing)."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print progress information."
    )
    return parser.parse_args()


def find_frames(frames_dir: Path, frame_exts: Iterable[str]) -> List[Path]:
    normalized_exts = {ext.lower() for ext in frame_exts}
    frames = [
        path
        for path in sorted(frames_dir.iterdir())
        if path.suffix.lower() in normalized_exts
    ]
    return frames


def load_mask(
    path: Path, size: tuple[int, int], threshold: int, dilation: int
) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    mask_array = np.asarray(mask)
    binary_mask = mask_array > threshold
    if dilation > 0:
        # Work with a binary image so morphological operations behave predictably.
        morph_image = Image.fromarray(binary_mask.astype(np.uint8) * 255, mode="L")
        kernel_size = dilation * 2 + 1
        morph_image = morph_image.filter(ImageFilter.MaxFilter(size=kernel_size))
        binary_mask = np.asarray(morph_image) > 0
    return binary_mask


def apply_combination(
    frame_array: np.ndarray,
    combined_mask: np.ndarray,
) -> np.ndarray:
    masked = np.full_like(frame_array, 255)
    masked[combined_mask] = frame_array[combined_mask]
    return masked


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {args.frames_dir}")
    missing_mask_dirs = [path for path in args.mask_dirs if not path.is_dir()]
    if missing_mask_dirs:
        raise FileNotFoundError(
            "Mask directories not found: "
            + ", ".join(str(path) for path in missing_mask_dirs)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = find_frames(args.frames_dir, args.frame_exts)
    if not frames:
        logging.info("No frames found in %s", args.frames_dir)
        return

    for frame_path in frames:
        frame_stem = frame_path.stem
        mask_glob = f"{frame_stem}_object_*.png"
        masks = []
        for mask_dir in args.mask_dirs:
            masks.extend(mask_dir.glob(mask_glob))
        masks = sorted(set(masks))
        if not masks:
            logging.info("No masks for frame %s", frame_stem)
            continue

        frame = Image.open(frame_path).convert("RGB")
        frame_array = np.asarray(frame)
        frame_size = frame.size

        combined_mask = np.zeros(frame_array.shape[:2], dtype=bool)
        for mask_path in masks:
            mask_array = load_mask(
                mask_path,
                frame_size,
                args.mask_threshold,
                args.dilation,
            )
            combined_mask |= mask_array

        if args.closing > 0:
            kernel_size = args.closing * 2 + 1
            morph_image = Image.fromarray(
                combined_mask.astype(np.uint8) * 255, mode="L"
            )
            morph_image = morph_image.filter(ImageFilter.MaxFilter(size=kernel_size))
            morph_image = morph_image.filter(ImageFilter.MinFilter(size=kernel_size))
            combined_mask = np.asarray(morph_image) > 0

        masked_frame = apply_combination(frame_array, combined_mask)
        output_name = f"{frame_stem}{frame_path.suffix}"
        output_path = args.output_dir / output_name
        Image.fromarray(masked_frame).save(output_path)

        logging.info("Processed %s with %d masks", frame_stem, len(masks))


if __name__ == "__main__":
    main()
