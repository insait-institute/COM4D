#!/bin/bash

source ~/.bashrc

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
echo "Activated com4d"

set -euo pipefail

COMMON_OUT_DIR=/work/berke_gokmen/data-1/com4d-out/
mkdir -p "$COMMON_OUT_DIR"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_key> [image_size]"
    exit 1
fi

CONFIG_KEY=$1
IMAGE_SIZE=${2:-518}
CONFIG_JSON=infer.json

if [ ! -f "${CONFIG_JSON}" ]; then
    echo "Error: Config file not found at ${CONFIG_JSON}"
    exit 1
fi

echo "Using configuration key: ${CONFIG_KEY}"
echo "Using conditioning image size: ${IMAGE_SIZE}"

python3 src/inference_com4d.py \
    --config_key "${CONFIG_KEY}" \
    --config_path "${CONFIG_JSON}" \
    --output_dir "${COMMON_OUT_DIR}" \
    --animation \
    --object_only_condition \
    --insert_rotation_every 20 \
    --prevent_collisions 0 \
    --scene_attn_ids "0,2,4,6,8,10,12,14,16,18,20" \
    --dynamic_attn_ids "0,1,3,5,7,9,11,13,15,17,19,21" \
    --base_weights_dir "pretrained_weights/TripoSG" \
    --image_size "${IMAGE_SIZE}"

