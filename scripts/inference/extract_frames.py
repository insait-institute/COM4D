#!/usr/bin/env python3
"""
extract_frames.py — Extract frames from a video and save as images.

Usage:
  python extract_frames.py /path/to/video.mp4 /path/to/output_dir \
      --ext png --stride 1 --start 0 --end -1 --resize 0x0
  
  python dataset_utils/extract_frames.py /work/berke_gokmen/data-1/assets/teaser_LAST/video_teaser.mp4 /work/berke_gokmen/data-1/assets/teaser_LAST/frames \
      --ext png --stride 1 --start 0 --resize 0x0
"""

import cv2
import argparse
from pathlib import Path

def parse_size(s: str):
    """Parse WxH like 1280x720 or 0x0 for no resize."""
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("Size must be like 1280x720 or 0x0")

def main():
    ap = argparse.ArgumentParser(description="Extract frames from a video.")
    ap.add_argument("video", type=Path, help="Path to input video file")
    ap.add_argument("outdir", type=Path, help="Directory to write frames")
    ap.add_argument("--ext", type=str, default="jpg", choices=["png","jpg","jpeg","bmp","webp"],
                    help="Image format/extension")
    ap.add_argument("--stride", type=int, default=1, help="Save every Nth frame (>=1)")
    ap.add_argument("--start", type=int, default=0, help="Start frame index (0-based)")
    ap.add_argument("--end", type=int, default=-1, help="End frame index (inclusive). -1 = till end")
    ap.add_argument("--resize", type=parse_size, default="0x0",
                    help="Resize to WxH; use 0x0 to keep original")
    args = ap.parse_args()

    print(args)

    print("[INFO] Starting frame extraction...")
    print(f"[INFO] Video: {args.video}")
    print(f"[INFO] Output Directory: {args.outdir}")
    print(f"[INFO] Image Format: {args.ext}")
    print(f"[INFO] Frame Stride: {args.stride}")
    print(f"[INFO] Start Frame: {args.start}")
    print(f"[INFO] End Frame: {args.end if args.end >= 0 else 'till end'}")
    print(f"[INFO] Resize: {args.resize[0]}x{args.resize[1]}")

    if args.stride < 1:
        ap.error("--stride must be >= 1")

    cap = cv2.VideoCapture(str(args.video))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end = total - 1 if args.end < 0 else args.end
    end = max(end, args.start)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Zero-padding length (e.g., 000123)
    pad = max(6, len(str(end)))

    # Fast-seek to start (best-effort; some codecs are non-accurate)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    frame_idx = args.start
    saved = 0
    target_w, target_h = args.resize

    print(f"[INFO] Video: {args.video}")
    print(f"[INFO] Frames: approx {total}, FPS: {fps:.3f}, Size: {width}x{height}")
    print(f"[INFO] Saving frames {args.start}..{end} stride={args.stride} -> {args.outdir}")

    while frame_idx <= end:
        ret, frame = cap.read()
        if not ret:
            break  # end or read error

        # Only save if this frame matches the stride
        if (frame_idx - args.start) % args.stride == 0:
            if target_w > 0 and target_h > 0:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # OpenCV uses BGR; most formats are fine with BGR
            fname = args.outdir / f"{frame_idx:0{pad}d}.{args.ext}"
            success = cv2.imwrite(str(fname), frame)
            if not success:
                print(f"[WARN] Failed to write {fname}")
            else:
                saved += 1

        frame_idx += 1

    cap.release()
    print(f"[DONE] Saved {saved} frames to {args.outdir}")

if __name__ == "__main__":
    main()