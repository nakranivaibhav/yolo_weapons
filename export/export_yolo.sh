#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="/workspace/yolo_dangerous_weapons/models/yolo/9_jan_2026_yolo11m/weights/best.pt"
OUTPUT_DIR=""
BATCH_SIZE=1
IMGSZ=640

EXPORT_VENV="${PROJECT_ROOT}/export/.venv"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model file not found at ${MODEL_PATH}"
    echo "Please edit the MODEL_PATH variable in this script."
    exit 1
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR:-same directory as model}"
echo "Batch size: ${BATCH_SIZE}"
echo "Image size: ${IMGSZ}"
echo "Export method: ONNX → trtexec (FP16)"
echo ""

if ! command -v trtexec &> /dev/null; then
    echo "⚠️  WARNING: trtexec not found in PATH"
    echo "The export script will fail unless trtexec is available."
    echo "Please ensure TensorRT is properly installed and trtexec is in PATH."
    echo ""
fi

cd "${SCRIPT_DIR}"

if [ ! -d "${EXPORT_VENV}" ]; then
    echo "Error: Virtual environment not found at ${EXPORT_VENV}"
    echo "Please ensure the venv exists before running this script."
    exit 1
fi

echo "Exporting YOLO model to TensorRT via trtexec..."
echo ""

if [ -n "${OUTPUT_DIR}" ]; then
    "${EXPORT_VENV}/bin/python" export_yolo.py "${MODEL_PATH}" "${OUTPUT_DIR}" "${BATCH_SIZE}" "${IMGSZ}"
else
    "${EXPORT_VENV}/bin/python" export_yolo.py "${MODEL_PATH}" "" "${BATCH_SIZE}" "${IMGSZ}"
fi
