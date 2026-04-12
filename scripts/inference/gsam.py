#!/usr/bin/env python3
"""
Track all humans in a folder of frames using Grounding DINO + SAM 2 and
save a binary mask for every detected person in every frame.

Each saved mask file follows the pattern `<frame_basename>_object_<idx>.png`.
"""

import argparse
import os
from typing import List, Sequence
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track humans across frames and export per-object masks."
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory containing ordered RGB frames (e.g. jpg, png).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store the generated mask images.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default="./checkpoints/sam2.1_hiera_large.pt",
        help="Path to the SAM 2 checkpoint file.",
    )
    parser.add_argument(
        "--sam2-config",
        default="sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to the SAM 2 model config.",
    )
    parser.add_argument(
        "--grounding-model-id",
        default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace model id for Grounding DINO.",
    )
    parser.add_argument(
        "--ann-frame-idx",
        type=int,
        default=0,
        help="Frame index used to initialize tracking (default: 0).",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.3,
        help="Text score threshold for Grounding DINO.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Box confidence threshold for detections.",
    )
    parser.add_argument(
        "--prompt",
        default="person.",
        help="Text prompt for human detection; must be lower case and end with a dot.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force computation device; defaults to CUDA if available.",
    )
    return parser.parse_args()


def natural_sort_key(name: str) -> Sequence:
    root, _ = os.path.splitext(name)
    key: List = []
    buffer = ""
    for ch in root:
        if ch.isdigit():
            buffer += ch
        else:
            if buffer:
                key.append(int(buffer))
                buffer = ""
            key.append(ch)
    if buffer:
        key.append(int(buffer))
    return key


def list_frame_paths(frames_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    frames = [
        name
        for name in os.listdir(frames_dir)
        if os.path.splitext(name)[-1].lower() in exts
    ]
    if not frames:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    frames.sort(key=natural_sort_key)
    return frames


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_device(sample, device: str):
    if isinstance(sample, (list, tuple)):
        return type(sample)(to_device(s, device) for s in sample)
    if hasattr(sample, "to"):
        return sample.to(device)
    return sample


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir
    output_dir = args.output_dir
    ensure_dir(output_dir)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_autocast = device == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else None
    if autocast_ctx is not None:
        autocast_ctx.__enter__()

    try:
        print("Building SAM2 from {}, {}".format(args.sam2_config, args.sam2_checkpoint))
        video_predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint)
        sam2_image_model = build_sam2(args.sam2_config, args.sam2_checkpoint)
        image_predictor = SAM2ImagePredictor(sam2_image_model)

        processor = AutoProcessor.from_pretrained(args.grounding_model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.grounding_model_id
        ).to(device)

        frame_names = list_frame_paths(frames_dir)
        ann_frame_idx = max(0, min(args.ann_frame_idx, len(frame_names) - 1))
        ann_frame_path = os.path.join(frames_dir, frame_names[ann_frame_idx])

        inference_state = video_predictor.init_state(video_path=frames_dir)

        init_image = Image.open(ann_frame_path).convert("RGB")
        inputs = processor(images=init_image, text=args.prompt, return_tensors="pt")
        inputs = to_device(inputs, device)

        with torch.no_grad():
            dino_outputs = grounding_model(**inputs)

        post_processed = processor.post_process_grounded_object_detection(
            dino_outputs,
            inputs["input_ids"],
            text_threshold=args.text_threshold,
            target_sizes=[init_image.size[::-1]],
        )

        detections = post_processed[0]
        if "boxes" not in detections or not len(detections["boxes"]):
            raise RuntimeError("Grounding DINO did not find any humans in the init frame.")

        boxes = detections["boxes"].cpu().numpy()
        scores = detections.get("scores", torch.ones(len(boxes)))
        keep: List[int] = [
            idx for idx, score in enumerate(scores) if float(score) >= args.box_threshold
        ]
        if not keep:
            raise RuntimeError(
                f"No detections remained after filtering with box_threshold {args.box_threshold}."
            )

        boxes = boxes[keep]
        labels = [detections["labels"][idx] for idx in keep]

        image_predictor.set_image(np.array(init_image))
        masks, _, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        if masks.ndim == 3:
            masks = masks[None]
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        for obj_id, mask in enumerate(masks, start=1):
            _, _, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                mask=mask.astype(np.float32),
            )

        video_segments = {}
        for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[frame_idx] = {
                int(obj_id): (mask_logits[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }

        for frame_idx, segments in video_segments.items():
            frame_name = frame_names[frame_idx]
            frame_root, _ = os.path.splitext(frame_name)
            for obj_id, mask in segments.items():
                mask_array = np.squeeze(np.asarray(mask))
                if mask_array.ndim != 2:
                    raise ValueError(
                        f"Unexpected mask shape {mask_array.shape} for object {obj_id} in frame {frame_idx}"
                    )
                mask_uint8 = (mask_array.astype(np.uint8)) * 255
                out_path = os.path.join(output_dir, f"{frame_root}_object_{obj_id:03d}.png")
                Image.fromarray(mask_uint8, mode="L").save(out_path)

        label_path = os.path.join(output_dir, "object_labels.txt")
        with open(label_path, "w", encoding="utf-8") as label_file:
            for obj_id, label in enumerate(labels, start=1):
                label_file.write(f"{obj_id}\t{label}\n")

        print(f"Saved masks for {len(video_segments)} frames to {output_dir}")
        print(f"Object id to label mapping stored at: {label_path}")
    finally:
        if autocast_ctx is not None:
            autocast_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
