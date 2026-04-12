#!/usr/bin/env python3
"""Batch-resize all images in a directory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize every image in a directory to the specified width and height.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing images to resize.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where resized images will be written.",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Target width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Target height in pixels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the output directory.",
    )

    args = parser.parse_args(argv)

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive integers")

    if not args.input_dir.is_dir():
        parser.error(f"input_dir '{args.input_dir}' is not a directory")

    return args


def iter_images(input_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    ]


def resize_image(src: Path, dst: Path, size: tuple[int, int], overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output file exists (use --overwrite to replace): {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as image:
        resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        resized.save(dst)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    size = (args.width, args.height)
    images = iter_images(args.input_dir)

    if not images:
        print(f"No images found in '{args.input_dir}'.", file=sys.stderr)
        return 1

    failures: list[str] = []
    for src in images:
        relative_path = src.relative_to(args.input_dir)
        dst = args.output_dir / relative_path

        # make dst have jpg extension
        dst = dst.with_suffix(".jpg")

        try:
            resize_image(src, dst, size, args.overwrite)
            print(f"Resized {src} -> {dst}")
        except FileExistsError as err:
            failures.append(str(err))
        except Exception as err:  # pragma: no cover - catch unexpected Pillow errors
            failures.append(f"Failed to resize {src}: {err}")

    if failures:
        print("\nEncountered errors:", file=sys.stderr)
        for msg in failures:
            print(f" - {msg}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
