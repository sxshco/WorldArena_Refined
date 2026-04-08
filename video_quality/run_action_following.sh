#!/bin/bash
set -euo pipefail

# Usage: run_action_following.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [CONFIG_PATH]

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
CONFIG_PATH=${4:-"$SCRIPT_DIR/config/config.yaml"}
WORLD_ARENA_PYTHON=${WORLD_ARENA_PYTHON:-$(command -v python3 || command -v python)}

if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ]; then
  echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [CONFIG_PATH]"
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

DATA_DIR="$SCRIPT_DIR/data_action_following"
OUTPUT_DIR_ACTION="$SCRIPT_DIR/output_action_following"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR_ACTION"
rm -rf "$DATA_DIR/gt_dataset" "$DATA_DIR/generated_dataset"

echo ">>> Running action_following preprocessing..."
"$WORLD_ARENA_PYTHON" preprocess_datasets_diversity.py --summary_json "$SUMMARY_JSON" --gen_video_dir "$GEN_VIDEO_DIR" --output_base "$DATA_DIR"

echo ">>> Running action_following evaluation..."
"$WORLD_ARENA_PYTHON" evaluate.py --dimension action_following --config "$CONFIG_PATH" --overwrite || echo ">>> [WARNING] evaluate.py (action_following) returned non-zero code"

echo ">>> ✅ Action Following Script Finished"
