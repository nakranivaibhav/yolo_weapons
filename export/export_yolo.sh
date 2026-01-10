#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${PROJECT_ROOT}/models/yolo/25_dec_2025_yolo11m/weights/best.pt"
OUTPUT_DIR="/workspace/exports"
BATCH_SIZE=1
IMGSZ=640

EXPORT_VENV="${SCRIPT_DIR}/.venv"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model file not found at ${MODEL_PATH}"
    echo "Please edit the MODEL_PATH variable in this script."
    exit 1
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR:-same as model}"
echo "Batch size: ${BATCH_SIZE}"
echo "Image size: ${IMGSZ}"
echo "Export method: ONNX → trtexec (FP16)"
echo ""

cd "${SCRIPT_DIR}"

if [ ! -d "${EXPORT_VENV}" ]; then
    echo "Creating export environment with uv (one-time setup)..."
    uv venv "${EXPORT_VENV}" --python 3.12
    uv pip install --python "${EXPORT_VENV}/bin/python" -r requirements-export.txt
    echo ""
fi

echo "Exporting YOLO model to TensorRT via trtexec..."
echo ""

if [ -n "${OUTPUT_DIR}" ]; then
    "${EXPORT_VENV}/bin/python" export_yolo.py "${MODEL_PATH}" "${OUTPUT_DIR}" "${BATCH_SIZE}" "${IMGSZ}"
else
    "${EXPORT_VENV}/bin/python" export_yolo.py "${MODEL_PATH}" "" "${BATCH_SIZE}" "${IMGSZ}"
fi
