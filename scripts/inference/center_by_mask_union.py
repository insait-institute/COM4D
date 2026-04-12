#!/usr/bin/env python3
"""
Center images based on the global union of segmentation masks.

Given one or more mask directories, we compute the union of all mask pixels
to find a bounding box and center for the subject across every frame. The
resulting translation offset is then applied to every image in the provided
image directories so that all assets share the same alignment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Center images by translating them using mask union offsets."
    )
    parser.add_argument(
        "--mask-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Mask directories whose contents will be unioned to determine offsets.",
    )
    parser.add_argument(
        "--image-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Image directories to translate using the computed offsets.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=0,
        help="Mask pixel values above this threshold are treated as foreground.",
    )
    parser.add_argument(
        "--fill",
        type=str,
        default="black",
        choices=["black", "white"],
        help="Background fill color when translating images.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


def list_image_paths(directories: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")
        files.extend(
            sorted(
                path
                for path in directory.iterdir()
                if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file()
            )
        )
    return files


def compute_union_bbox(
    mask_paths: Sequence[Path], threshold: int
) -> Tuple[int, int, int, int, Tuple[int, int]]:
    min_x: Optional[int] = None
    min_y: Optional[int] = None
    max_x: Optional[int] = None
    max_y: Optional[int] = None
    image_size: Optional[Tuple[int, int]] = None

    for mask_path in mask_paths:
        with Image.open(mask_path) as mask_img:
            mask_gray = mask_img.convert("L")
            if image_size is None:
                image_size = mask_gray.size
            elif mask_gray.size != image_size:
                raise ValueError(
                    f"Mask {mask_path} has size {mask_gray.size}, "
                    f"expected {image_size}"
                )

            mask_array = np.asarray(mask_gray)
            foreground = mask_array > threshold
            if not foreground.any():
                continue

            ys, xs = np.nonzero(foreground)
            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max())
            y1 = int(ys.max())

            min_x = x0 if min_x is None else min(min_x, x0)
            min_y = y0 if min_y is None else min(min_y, y0)
            max_x = x1 if max_x is None else max(max_x, x1)
            max_y = y1 if max_y is None else max(max_y, y1)

    if image_size is None:
        raise ValueError("No masks were found in the provided directories.")

    if min_x is None or min_y is None or max_x is None or max_y is None:
        raise ValueError("Masks contained no foreground pixels to compute a union.")

    return min_x, min_y, max_x, max_y, image_size


def compute_offset(
    bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]
) -> Tuple[int, int]:
    min_x, min_y, max_x, max_y = bbox
    width, height = image_size

    bbox_center_x = (min_x + max_x) / 2.0
    bbox_center_y = (min_y + max_y) / 2.0
    image_center_x = (width - 1) / 2.0
    image_center_y = (height - 1) / 2.0

    shift_x = int(round(image_center_x - bbox_center_x))
    shift_y = int(round(image_center_y - bbox_center_y))
    return shift_x, shift_y


def translate_image(
    image: Image.Image, shift_x: int, shift_y: int, fill_value: int
) -> Image.Image:
    array = np.asarray(image)
    shifted = shift_array(array, shift_x, shift_y, fill_value)
    return Image.fromarray(shifted, mode=image.mode)


def shift_array(
    array: np.ndarray, shift_x: int, shift_y: int, fill_value: int
) -> np.ndarray:
    height, width = array.shape[:2]
    shifted = np.full_like(array, fill_value)

    x_overlap = compute_overlap(width, shift_x)
    y_overlap = compute_overlap(height, shift_y)

    if x_overlap is None or y_overlap is None:
        return shifted

    src_x0, src_x1, dst_x0, dst_x1 = x_overlap
    src_y0, src_y1, dst_y0, dst_y1 = y_overlap

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return shifted


def compute_overlap(length: int, shift: int) -> Optional[Tuple[int, int, int, int]]:
    if shift >= 0:
        src_start = 0
        src_end = length - shift
        dst_start = shift
    else:
        src_start = -shift
        src_end = length
        dst_start = 0

    overlap = src_end - src_start
    if overlap <= 0:
        return None

    dst_end = dst_start + overlap
    return src_start, src_end, dst_start, dst_end


def process_images(
    image_dirs: Sequence[Path], shift_x: int, shift_y: int, fill_value: int
) -> None:
    image_paths = list_image_paths(image_dirs)
    if not image_paths:
        logging.warning("No images found in the provided image directories.")
        return

    for path in image_paths:
        with Image.open(path) as img:
            translated = translate_image(img, shift_x, shift_y, fill_value)
            translated.save(path)
        logging.info("Centered image saved to %s", path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    mask_paths = list_image_paths(args.mask_dirs)
    if not mask_paths:
        raise FileNotFoundError("No mask files found in the provided directories.")

    bbox = compute_union_bbox(mask_paths, args.mask_threshold)
    shift_x, shift_y = compute_offset(bbox[:4], bbox[4])

    logging.info("Union bounding box: %s", bbox[:4])
    logging.info("Image size: %s", bbox[4])
    logging.info("Computed shift (x, y): (%d, %d)", shift_x, shift_y)

    fill_value = 255 if args.fill == "white" else 0
    process_images(args.image_dirs, shift_x, shift_y, fill_value)


if __name__ == "__main__":
    main()
