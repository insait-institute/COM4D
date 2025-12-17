#!/bin/bash

# --- ensure micromamba env; if activation fails, run env.sh then retry ---
ensure_env() {
    # load micromamba shell hook if present
    if command -v micromamba >/dev/null 2>&1; then
        eval "$(micromamba shell hook -s bash)" || true
    fi

    # try to activate; on failure, run env.sh and retry once
    if ! micromamba activate com4d >/dev/null 2>&1; then
        echo "[setup] activation failed for 'com4d'. Running env.sh then retrying ..."
        bash scripts/env.sh || { echo "[setup] env.sh failed"; exit 1; }
        eval "$(micromamba shell hook -s bash)" || true
        micromamba activate com4d || { echo "[setup] activation still failing"; exit 1; }
    fi
}

ensure_env

micromamba activate com4d

NUM_MACHINES=1
NUM_LOCAL_GPUS=1
MACHINE_RANK=0
CONFIG_NAME=1_mf8_mp8_nt512

OUT_DIR="#PATH"
PRETRAINED_MODEL_PATH="#PATH"
PRETRAINED_MODEL_CKPT="#NUMBER"

mkdir -p $OUT_DIR

export WANDB_API_KEY="#WANDB_KEY" # Modify this if you use wandb

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    src/train_com4d.py \
        --pin_memory \
        --allow_tf32 \
        --config configs/${CONFIG_NAME}.yaml --use_ema \
        --gradient_accumulation_steps 4 \
        --output_dir $OUT_DIR \
        --tag ${CONFIG_NAME} \
        --val_only_rank0 \
        # --load_pretrained_model $PRETRAINED_MODEL_PATH \
        # --load_pretrained_model_ckpt $PRETRAINED_MODEL_CKPT \
        # --offline_wandb
        
        