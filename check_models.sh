#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

echo "=============================================="
echo "   Model Files Check (Git LFS)"
echo "=============================================="
echo ""

all_good=true

check_file() {
    local file="$1"
    local name="$2"
    
    if [ ! -f "$file" ]; then
        echo "  ✗ $name: NOT FOUND"
        all_good=false
        return
    fi
    
    # Check if first line contains Git LFS marker
    first_line=$(head -1 "$file" 2>/dev/null)
    if [[ "$first_line" == "version https://git-lfs.github.com"* ]]; then
        expected_size=$(grep "^size " "$file" | awk '{print $2}')
        if [ -n "$expected_size" ]; then
            expected_mb=$(echo "scale=1; $expected_size / 1024 / 1024" | bc)
            echo "  ✗ $name: Git LFS pointer (expecting ${expected_mb}MB)"
        else
            echo "  ✗ $name: Git LFS pointer (not downloaded)"
        fi
        all_good=false
    else
        size=$(du -h "$file" | cut -f1)
        echo "  ✓ $name: $size (actual file)"
    fi
}

echo "[1/3] Checking YOLO models..."
check_file "models/yolo/weapon_detection_yolo11m_640/weights/best.pt" "YOLO11m PT model"
check_file "models/yolo/weapon_detection_yolo11m_640/weights/best_fp16.engine" "YOLO11m FP16 engine"
check_file "models/yolo/weapon_detection_yolo11m_640/weights/best_fp32.engine" "YOLO11m FP32 engine"
check_file "models/yolo/weapon_detection_yolo11m_640/weights/best_int8.engine" "YOLO11m INT8 engine"
echo ""

echo "[2/3] Checking YOLO11s models..."
check_file "models/yolo/weapon_detection_yolo11s_1280/weights/best.pt" "YOLO11s PT model"
check_file "models/yolo/weapon_detection_yolo11s_1280/weights/best_fp16.engine" "YOLO11s FP16 engine"
check_file "models/yolo/weapon_detection_yolo11s_1280/weights/best_fp32.engine" "YOLO11s FP32 engine"
check_file "models/yolo/weapon_detection_yolo11s_1280/weights/best_int8.engine" "YOLO11s INT8 engine"
echo ""

echo "[3/3] Checking ConvNeXT model..."
check_file "models/convnext_compiled/convnext_bs4.pt" "ConvNeXT compiled model"
echo ""

echo "=============================================="
if [ "$all_good" = true ]; then
    echo "✅ All models are properly downloaded!"
    echo ""
    echo "You can now run inference:"
    echo "  cd inference && ./tiled_run.sh"
else
    echo "⚠️  Some models are missing or not downloaded"
    echo ""
    echo "To download all models, run:"
    echo "  sudo apt-get install git-lfs"
    echo "  git lfs install"
    echo "  git lfs pull"
    echo ""
    echo "Or to download specific models:"
    echo "  git lfs pull --include=\"models/yolo/**/*.engine\""
    echo "  git lfs pull --include=\"models/convnext_compiled/*.pt\""
fi
echo "=============================================="
echo ""

