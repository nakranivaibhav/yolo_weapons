#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${PROJECT_ROOT}/models/yolo/weapon_detection_yolo11m_640/weights/best.pt"
DATA_YAML="${PROJECT_ROOT}/data/yolo_dataset/data.yaml"

if [ $# -ge 1 ]; then
    MODEL_PATH="$1"
fi

if [ $# -ge 2 ]; then
    DATA_YAML="$2"
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Model path: ${MODEL_PATH}"
echo "Data YAML: ${DATA_YAML}"
echo ""

if [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model file not found at ${MODEL_PATH}"
    echo ""
    echo "Usage: $0 [model_path] [data_yaml]"
    echo "Example: $0 ../models/yolo/weapon_detection_yolo11m_640/weights/best.pt"
    exit 1
fi

cd "${SCRIPT_DIR}"

echo "Exporting YOLO model to TensorRT..."
echo ""

uv run python export_yolo.py "${MODEL_PATH}" "${DATA_YAML}"
