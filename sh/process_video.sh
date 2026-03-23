#!/bin/bash

# Update the following parameters
SAM2_CHECKPOINT="/work/berke_gokmen/ckpts/sam2.1_hiera_large.pt"
SAM2_CONFIG="//home/berke_gokmen/project/COM4D/sam2/sam2_hiera_l.yaml" # You might need to put // before the path, ex: //home/berke_gokmen/project/COM4D/sam2/sam2_hiera_l.yaml

VIDEO_PATH=$1
START=${2:-0}
END=${3:-100}
MASK_PROMPT=${4:-"person."}
MASK_STATIC_PROMPT=${5:-""} # e.g., "table., chair., tree."

echo "Processing video: $VIDEO_PATH from frame $START to $END with mask prompt '$MASK_PROMPT' and static mask prompt '$MASK_STATIC_PROMPT'"

VIDEO_NAME=$(basename "$VIDEO_PATH" | cut -d. -f1)
OUTPUT_PATH=assets/${VIDEO_NAME}
FRAMES_PATH=${OUTPUT_PATH}/frames
FRAMES_NO_BG_PATH=${OUTPUT_PATH}/frames_no_bg
MASKS_PATH=${OUTPUT_PATH}/masks
MASKS_STATIC_PATH=${OUTPUT_PATH}/masks_static

mkdir -p $FRAMES_PATH
mkdir -p $FRAMES_NO_BG_PATH
mkdir -p $MASKS_PATH
mkdir -p $MASKS_STATIC_PATH

echo "Extracting frames from video..."
python scripts/inference/extract_frames.py $VIDEO_PATH $FRAMES_PATH --start $START --end $END

echo "Padding frames to square dimensions..."
python scripts/inference/pad_to_square.py \
  --input_dir "$FRAMES_PATH" \
  --output_dir "$FRAMES_PATH"

echo "Resizing frames to 768x768"
python scripts/inference/resize_images.py $FRAMES_PATH $FRAMES_PATH \
    --height 768 --width 768 --overwrite

echo "Extracting masks using Grounded-SAM... Prompt: $MASK_PROMPT"
python scripts/inference/gsam.py \
  --frames-dir "$FRAMES_PATH" \
  --output-dir "$MASKS_PATH" \
  --sam2-checkpoint "$SAM2_CHECKPOINT" \
  --sam2-config "$SAM2_CONFIG" \
  --prompt "$MASK_PROMPT" \
  --box-threshold 0.5 \
  --text-threshold 0.4

if [[ -n "$MASK_STATIC_PROMPT" ]]; then
  echo "Extracting static masks using Grounded-SAM-2... Prompt: $MASK_STATIC_PROMPT"
  python scripts/inference/gsam.py \
    --frames-dir "$FRAMES_PATH" \
    --output-dir "$MASKS_STATIC_PATH" \
    --sam2-checkpoint "$SAM2_CHECKPOINT" \
    --sam2-config "$SAM2_CONFIG" \
    --prompt "$MASK_STATIC_PROMPT" \
    --box-threshold 0.35 \
    --text-threshold 0.4
else
  echo "Skipping static mask extraction because static mask prompt is empty."
fi

echo "Combining masks with frames..."
python -u scripts/inference/apply_mask_combinations.py \
  "$FRAMES_PATH" "$MASKS_PATH" "$MASKS_STATIC_PATH" "$FRAMES_NO_BG_PATH" --dilation 0 --closing 3

echo "Centering frames and masks using the union of masks..."
python scripts/inference/center_by_mask_union.py \
  --mask-dirs "$MASKS_PATH" "$MASKS_STATIC_PATH" \
  --image-dirs "$FRAMES_NO_BG_PATH" "$FRAMES_PATH" \
  --fill white
    
python scripts/inference/center_by_mask_union.py \
  --mask-dirs "$MASKS_PATH" "$MASKS_STATIC_PATH" \
  --image-dirs "$MASKS_PATH" "$MASKS_STATIC_PATH" \
  --fill black
