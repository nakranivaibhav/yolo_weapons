#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${PROJECT_ROOT}/models/convnext_trained/best_checkpoint"
OUTPUT_DIR="${PROJECT_ROOT}/models/convnext_compiled"
BATCH_SIZE=4
INPUT_SIZE=224
MODE="reduce-overhead"

echo "Project root: ${PROJECT_ROOT}"
echo "Model path: ${MODEL_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

cd "${SCRIPT_DIR}"

export LD_LIBRARY_PATH="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/tensorrt_libs:$LD_LIBRARY_PATH"

uv run python export_convnext.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --input_size $INPUT_SIZE \
    --mode $MODE \
    --benchmark

