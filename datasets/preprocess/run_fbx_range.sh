#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_fbx_range.sh FBX_LIST OUTPUT_DIR BLENDER_BIN START_IDX END_IDX [WORKER_ARGS...]

Runs infer_utils/fbx_to_glb.py for each FBX listed between START_IDX and END_IDX
(inclusive, zero-based) in FBX_LIST. Additional WORKER_ARGS are forwarded to the
worker script (e.g. --start-frame 10 --end-frame 120 --frame-step 2).
EOF
}

if [[ $# -lt 5 ]]; then
    usage >&2
    exit 1
fi

FBX_LIST=$(realpath "$1")
OUTPUT_DIR=$(realpath "$2")
BLENDER_BIN="$3"
START_IDX="$4"
END_IDX="$5"
shift 5
EXTRA_ARGS=("$@")

if [[ ! -f "$FBX_LIST" ]]; then
    echo "FBX list file not found: $FBX_LIST" >&2
    exit 1
fi

if [[ ! -x "$BLENDER_BIN" ]]; then
    echo "Blender binary not executable: $BLENDER_BIN" >&2
    exit 1
fi

if ! [[ "$START_IDX" =~ ^[0-9]+$ && "$END_IDX" =~ ^[0-9]+$ ]]; then
    echo "START_IDX and END_IDX must be non-negative integers." >&2
    exit 1
fi

if (( START_IDX > END_IDX )); then
    echo "START_IDX ($START_IDX) cannot be greater than END_IDX ($END_IDX)." >&2
    exit 1
fi

mapfile -t FBX_PATHS < "$FBX_LIST"
TOTAL=${#FBX_PATHS[@]}

if (( TOTAL == 0 )); then
    echo "FBX list is empty: $FBX_LIST" >&2
    exit 1
fi

if (( START_IDX >= TOTAL || END_IDX >= TOTAL )); then
    echo "Index range [$START_IDX, $END_IDX] exceeds list length ($TOTAL)." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
WORKER_SCRIPT="$SCRIPT_DIR/fbx_to_glb.py"

for (( idx = START_IDX; idx <= END_IDX; idx++ )); do
    FBX_PATH="${FBX_PATHS[idx]}"
    FBX_PATH="${FBX_PATH#"${FBX_PATH%%[![:space:]]*}"}"
    FBX_PATH="${FBX_PATH%"${FBX_PATH##*[![:space:]]}"}"

    if [[ -z "$FBX_PATH" ]]; then
        echo "[SKIP] Empty FBX path at index $idx" >&2
        continue
    fi

    if [[ ! -f "$FBX_PATH" ]]; then
        echo "[WARN] FBX file not found at index $idx: $FBX_PATH" >&2
        continue
    fi

    ABS_FBX=$(realpath "$FBX_PATH")
    echo "[RUN] idx=$idx file=$ABS_FBX" >&2

    "$BLENDER_BIN" -b --python "$WORKER_SCRIPT" -- \
        --fbx "$ABS_FBX" \
        --output-dir "$OUTPUT_DIR" \
        --no-progress \
        "${EXTRA_ARGS[@]}"
done

echo "Done processing indices $START_IDX through $END_IDX (list size $TOTAL)." >&2
