#!/usr/bin/env python3
"""Pad images with white margins so they become squares."""

import argparse
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageColor, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(folder: Path) -> List[Path]:
    files: Iterable[Path] = filter(Path.is_file, folder.iterdir())
    image_files = [path for path in files if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(image_files)


def white_fill(image: Image.Image):
    """Return a white color tuple compatible with the image mode."""
    try:
        return ImageColor.getcolor("white", image.mode)
    except ValueError:
        return tuple(255 for _ in image.getbands())


def pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image.copy()

    size = max(width, height)
    delta_w = size - width
    delta_h = size - height
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    fill = white_fill(image)
    return ImageOps.expand(image, border=padding, fill=fill)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pad images with white borders so they become squares."
    )
    parser.add_argument("--input_dir", type=Path, help="Directory containing source images.")
    parser.add_argument("--output_dir", type=Path, help="Directory to write padded images.")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive) of images to process after sorting (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive) of images to process after sorting (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(input_dir)
    if not image_files:
        raise ValueError(f"No supported image files found in {input_dir}")

    start_index = args.start
    end_index = args.end if args.end is not None else len(image_files)
    if start_index < 0 or start_index > len(image_files):
        raise ValueError(f"Start index {start_index} is out of range for {len(image_files)} files.")
    if end_index < start_index or end_index > len(image_files):
        raise ValueError(f"End index {end_index} is out of range for {len(image_files)} files.")

    selected_files = image_files[start_index:end_index]
    for path in selected_files:
        with Image.open(path) as img:
            padded = pad_to_square(img)
            padded.save(output_dir / path.name)


if __name__ == "__main__":
    main()
