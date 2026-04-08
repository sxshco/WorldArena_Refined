#!/bin/bash
set -euo pipefail
# Usage: run_evaluation_JEPA.sh <GEN_VIDEO_DIR> <REAL_VIDEO_DIR>

GEN_VIDEO_DIR=${1:-}
ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REAL_VIDEO_DIR=${2:-}
WORLD_ARENA_JEPA_ENV=${WORLD_ARENA_JEPA_ENV:-WorldArena_JEPA}

if [[ -z "$GEN_VIDEO_DIR" || -z "$REAL_VIDEO_DIR" ]]; then
  echo "Usage: $0 <GEN_VIDEO_DIR> <REAL_VIDEO_DIR>"
  exit 1
fi

OUTPUT_ROOT="$ROOT_DIR/output_JEDi"

cd "$ROOT_DIR/JEDi"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORLD_ARENA_JEPA_ENV"

python batch.py \
	--real_dir "$REAL_VIDEO_DIR" \
	--gen_dir "$GEN_VIDEO_DIR" \
    --output_root "$OUTPUT_ROOT" 
