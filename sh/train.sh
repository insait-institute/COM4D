#!/bin/bash

nvidia-smi

# --- ensure micromamba env; if activation fails, run env.sh then retry ---
ensure_env() {
    # load micromamba shell hook if present
    if command -v micromamba >/dev/null 2>&1; then
        eval "$(micromamba shell hook -s bash)" || true
    fi

    # try to activate; on failure, run env.sh and retry once
    if ! micromamba activate com4d >/dev/null 2>&1; then
        echo "[setup] activation failed for 'com4d'. Running env.sh then retrying ..."
        bash sh/env.sh || { echo "[setup] env.sh failed"; exit 1; }
        eval "$(micromamba shell hook -s bash)" || true
        micromamba activate com4d || { echo "[setup] activation still failing"; exit 1; }
    fi
}

# bash sh/env.sh
ensure_env

micromamba activate com4d

NUM_MACHINES=1
NUM_LOCAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
MACHINE_RANK=0
OUT_DIR=/work/berke_gokmen/ckpts/com4d_1_new
RANDOM=$$$(date +%s)  # generate a random seed based on current time and process ID

mkdir -p $OUT_DIR

export WANDB_API_KEY="" # Modify this if you use wandb

configs=(
    sdemb/mf8_mp8_nt512 # 0
    sdemb/mf16_mp16/mf8_mp8_nt512 # 1
    sdemb/mf16_mp16/dfot/mf8_mp8_nt512 # 2
    sdemb/mf16_mp16/dfot/mask/mf8_mp8_nt512 # 3
)

pretrained_model_paths=(
    /
    /
    /
    /
)

pretrained_model_ckpts=(
    -1
    -1
    -1
    -1
)

# give an error if no index is provided

if [ -z "$1" ]; then
    echo "Error: No configuration index provided."
    exit 1
fi

config_index=${1}  # Use first argument as config index, default to 0 if not provided

config_name=${configs[$config_index]}
pretrained_model_path=${pretrained_model_paths[$config_index]}
pretrained_model_ckpt=${pretrained_model_ckpts[$config_index]}

echo "Using configuration index: $config_index, config name: $config_name"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    src/train_com4d.py \
        --pin_memory \
        --allow_tf32 \
        --config configs/${config_name}.yaml --use_ema \
        --gradient_accumulation_steps 4 \
        --output_dir $OUT_DIR \
        --tag com4d_${config_name//\//_}_${RANDOM} \
        --val_only_rank0 \
        --load_pretrained_model $pretrained_model_path \
        --load_pretrained_model_ckpt $pretrained_model_ckpt \
        # --offline_wandb
        
        