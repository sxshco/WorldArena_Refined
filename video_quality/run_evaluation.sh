#!/bin/bash
set -euo pipefail

# Usage: run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH]
# METRIC_LIST example: "image_quality,photometric_smoothness,motion_smoothness"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
RAW_METRICS=${4:-}
CONFIG_PATH=${5:-"$SCRIPT_DIR/config/config.yaml"}
WORLD_ARENA_PYTHON=${WORLD_ARENA_PYTHON:-$(command -v python3 || command -v python)}

if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ] || [ -z "$RAW_METRICS" ]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH]"
    exit 1
fi

if [ ! -x "$WORLD_ARENA_PYTHON" ]; then
    echo ">>> [ERROR] WorldArena python not found: $WORLD_ARENA_PYTHON"
    echo ">>> Set WORLD_ARENA_PYTHON to your environment python if needed."
    exit 1
fi

if [ ! -f "$SUMMARY_JSON" ]; then
    echo ">>> [ERROR] summary.json not found: $SUMMARY_JSON"
    exit 1
fi

if [ ! -d "$GEN_VIDEO_DIR" ]; then
    echo ">>> [ERROR] generated video directory not found: $GEN_VIDEO_DIR"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo ">>> [ERROR] config file not found: $CONFIG_PATH"
    exit 1
fi

CLEAN_METRICS=$(echo "$RAW_METRICS" | tr ',' ' ' | tr '"' ' ')
METRIC_ARRAY=($CLEAN_METRICS)
echo ">>> Input metrics: $RAW_METRICS"
echo ">>> Formatted for evaluate.py: ${METRIC_ARRAY[*]}"

EVAL_METRICS=()
for metric in "${METRIC_ARRAY[@]}"; do
    if [ "$metric" == "action_following" ]; then
        echo ">>> [ERROR] action_following is handled by run_action_following.sh, not run_evaluation.sh"
        exit 1
    fi
    EVAL_METRICS+=("$metric")
done

if [ ${#EVAL_METRICS[@]} -eq 0 ]; then
    echo ">>> [ERROR] No standard metrics provided"
    exit 1
fi

DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
rm -rf "$DATA_DIR/gt_dataset" "$DATA_DIR/generated_dataset"

"$WORLD_ARENA_PYTHON" - "$CONFIG_PATH" "${EVAL_METRICS[@]}" <<'PY2'
import os
import sys
from pathlib import Path
import yaml

config_path = sys.argv[1]
metrics = sys.argv[2:]
config_path = str(Path(config_path).resolve())
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def resolve_path(value):
    if not isinstance(value, str):
        return value
    value = os.path.expanduser(os.path.expandvars(value))
    if value.startswith("./") or value.startswith("../"):
        return str((Path(config_path).parent / value).resolve())
    return value

required_by_metric = {
    'image_quality': [('ckpt', 'image_quality', 'musiq')],
    'photometric_smoothness': [('ckpt', 'photometric_smoothness', 'cfg'), ('ckpt', 'photometric_smoothness', 'model')],
    'motion_smoothness': [('ckpt', 'motion_smoothness', 'model')],
}

for metric in metrics:
    for key_path in required_by_metric.get(metric, []):
        node = config
        for key in key_path:
            node = node.get(key) if isinstance(node, dict) else None
        node = resolve_path(node)
        if not node:
            raise SystemExit(f"[ERROR] Missing config entry for {metric}: {'/'.join(key_path)}")
        if not os.path.exists(node):
            raise SystemExit(f"[ERROR] Required path for {metric} does not exist: {node}")

print('>>> Config preflight passed')
PY2

echo ">>> Running preprocessing for standard metrics..."
"$WORLD_ARENA_PYTHON" preprocess_datasets.py --summary_json "$SUMMARY_JSON" --gen_video_dir "$GEN_VIDEO_DIR" --output_base "$DATA_DIR"

echo ">>> Running processing (resize and detection/tracking)..."
"$WORLD_ARENA_PYTHON" ./processing/video_resize.py --config_path "$CONFIG_PATH"
"$WORLD_ARENA_PYTHON" ./processing/detection_tracking.py --config_path "$CONFIG_PATH" --detect_gt

echo ">>> Starting standard evaluation: ${EVAL_METRICS[*]}"
"$WORLD_ARENA_PYTHON" evaluate.py --dimension "${EVAL_METRICS[@]}" --config "$CONFIG_PATH" --overwrite

echo ">>> ✅ All standard evaluations finished"
