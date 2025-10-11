#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
DATA_DIR="${PROJECT_ROOT}/data"

echo "=============================================="
echo "   Dangerous Weapons Detection Dataset Setup"
echo "=============================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Data directory: ${DATA_DIR}"
echo ""

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

echo "[1/2] Downloading YOLO detection dataset..."
YOLO_FILE_ID="1HwUmZmDNpSyigVIBbRxDPn2xUQpLxBty"
YOLO_OUTPUT="yolo_dataset.tar.gz"

if [ ! -d "yolo_dataset" ]; then
    if [ ! -f "${YOLO_OUTPUT}" ]; then
        echo "  Downloading from Google Drive (ID: ${YOLO_FILE_ID})..."
        echo "  This is a large file (~2GB), please wait..."
        
        gdown --fuzzy "https://drive.google.com/file/d/${YOLO_FILE_ID}/view?usp=drive_link" -O "${YOLO_OUTPUT}"
        
        if [ $? -ne 0 ]; then
            echo ""
            echo "  ✗ Download failed with gdown. Trying alternative method..."
            rm -f "${YOLO_OUTPUT}"
            
            gdown "${YOLO_FILE_ID}" -O "${YOLO_OUTPUT}"
            
            if [ $? -ne 0 ]; then
                echo ""
                echo "  ✗ Automatic download failed."
                echo ""
                echo "  Please download manually:"
                echo "    1. Visit: https://drive.google.com/file/d/${YOLO_FILE_ID}/view?usp=drive_link"
                echo "    2. Download the file"
                echo "    3. Save as: ${DATA_DIR}/${YOLO_OUTPUT}"
                echo "    4. Run this script again"
                exit 1
            fi
        fi
        
        echo "  Verifying download..."
        if [ ! -s "${YOLO_OUTPUT}" ]; then
            echo "  ✗ Downloaded file is empty"
            rm -f "${YOLO_OUTPUT}"
            exit 1
        fi
        
        file_type=$(file -b "${YOLO_OUTPUT}")
        if [[ ! "${file_type}" =~ "gzip" ]] && [[ ! "${file_type}" =~ "compressed" ]]; then
            echo "  ✗ Downloaded file is not a valid tar.gz archive"
            echo "  File type detected: ${file_type}"
            echo ""
            echo "  Please download manually from:"
            echo "  https://drive.google.com/file/d/${YOLO_FILE_ID}/view?usp=drive_link"
            rm -f "${YOLO_OUTPUT}"
            exit 1
        fi
        
        echo "  ✓ Download complete and verified"
    else
        echo "  ✓ Already downloaded: ${YOLO_OUTPUT}"
    fi
    
    echo "  Extracting ${YOLO_OUTPUT}..."
    tar -xzf "${YOLO_OUTPUT}" 2>&1 | grep -v "Ignoring unknown extended header keyword" || true
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ✗ Extraction failed"
        echo "  The tar.gz file may be corrupted. Please download again manually."
        rm -f "${YOLO_OUTPUT}"
        exit 1
    fi
    
    echo "  ✓ Extraction complete"
    
    echo "  Cleaning up macOS metadata files..."
    find yolo_dataset -name "._*" -type f -delete 2>/dev/null || true
    find yolo_dataset -name ".DS_Store" -type f -delete 2>/dev/null || true
    echo "  ✓ Cleanup complete"
else
    echo "  ✓ Already extracted: yolo_dataset/"
fi

echo ""
echo "[2/2] Downloading ConvNeXT classification dataset..."
CONVNEXT_FILE_ID="1IRommjmeYrKsy0K5qLlrUR309HZfv5OY"
CONVNEXT_OUTPUT="yolo_dataset_cls_cropped.zip"

if [ ! -d "yolo_dataset_cls_cropped" ]; then
    if [ ! -f "${CONVNEXT_OUTPUT}" ]; then
        echo "  Downloading from Google Drive (ID: ${CONVNEXT_FILE_ID})..."
        
        gdown --fuzzy "https://drive.google.com/file/d/${CONVNEXT_FILE_ID}/view?usp=sharing" -O "${CONVNEXT_OUTPUT}"
        
        if [ $? -ne 0 ]; then
            echo ""
            echo "  ✗ Download failed with gdown. Trying alternative method..."
            rm -f "${CONVNEXT_OUTPUT}"
            
            gdown "${CONVNEXT_FILE_ID}" -O "${CONVNEXT_OUTPUT}"
            
            if [ $? -ne 0 ]; then
                echo ""
                echo "  ✗ Automatic download failed."
                echo ""
                echo "  Please download manually:"
                echo "    1. Visit: https://drive.google.com/file/d/${CONVNEXT_FILE_ID}/view?usp=sharing"
                echo "    2. Download the file"
                echo "    3. Save as: ${DATA_DIR}/${CONVNEXT_OUTPUT}"
                echo "    4. Run this script again"
                exit 1
            fi
        fi
        
        echo "  Verifying download..."
        if [ ! -s "${CONVNEXT_OUTPUT}" ]; then
            echo "  ✗ Downloaded file is empty"
            rm -f "${CONVNEXT_OUTPUT}"
            exit 1
        fi
        
        file_type=$(file -b "${CONVNEXT_OUTPUT}" | head -c 3)
        if [ "${file_type}" != "Zip" ]; then
            echo "  ✗ Downloaded file is not a valid ZIP archive"
            echo "  File type detected: $(file -b ${CONVNEXT_OUTPUT})"
            echo ""
            echo "  Please download manually from:"
            echo "  https://drive.google.com/file/d/${CONVNEXT_FILE_ID}/view?usp=sharing"
            rm -f "${CONVNEXT_OUTPUT}"
            exit 1
        fi
        
        echo "  ✓ Download complete and verified"
    else
        echo "  ✓ Already downloaded: ${CONVNEXT_OUTPUT}"
    fi
    
    echo "  Extracting ${CONVNEXT_OUTPUT}..."
    unzip -q "${CONVNEXT_OUTPUT}"
    
    if [ $? -ne 0 ]; then
        echo "  ✗ Extraction failed"
        echo "  The ZIP file may be corrupted. Please download again manually."
        rm -f "${CONVNEXT_OUTPUT}"
        exit 1
    fi
    
    echo "  ✓ Extraction complete"
    
    echo "  Cleaning up macOS metadata files..."
    find yolo_dataset_cls_cropped -name "._*" -type f -delete 2>/dev/null || true
    find yolo_dataset_cls_cropped -name ".DS_Store" -type f -delete 2>/dev/null || true
    find yolo_dataset_cls_cropped -name "__MACOSX" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "  ✓ Cleanup complete"
    
    if [ -d "yolo_dataset_cls_cropped" ]; then
        echo "  Creating symlink: convnext_dataset -> yolo_dataset_cls_cropped"
        ln -sf yolo_dataset_cls_cropped convnext_dataset
    fi
else
    echo "  ✓ Already extracted: yolo_dataset_cls_cropped/"
    
    if [ ! -L "convnext_dataset" ]; then
        echo "  Creating symlink: convnext_dataset -> yolo_dataset_cls_cropped"
        ln -sf yolo_dataset_cls_cropped convnext_dataset
    fi
fi

cd "${PROJECT_ROOT}"

echo ""
echo "=============================================="
echo "   Dataset Structure"
echo "=============================================="
echo ""
echo "YOLO Dataset (Object Detection):"
echo "  ${DATA_DIR}/yolo_dataset/"
echo "    ├── images/"
echo "    │   ├── train/"
echo "    │   ├── valid/"
echo "    │   └── test/"
echo "    ├── labels/"
echo "    │   ├── train/"
echo "    │   ├── valid/"
echo "    │   └── test/"
echo "    └── data.yaml"
echo ""
echo "ConvNeXT Dataset (Image Classification):"
echo "  ${DATA_DIR}/yolo_dataset_cls_cropped/ (aliased as convnext_dataset)"
echo "    ├── train/"
echo "    │   ├── gun/"
echo "    │   └── knife/"
echo "    └── valid/"
echo "        ├── gun/"
echo "        └── knife/"
echo ""
echo "Note: 'convnext_dataset' is a symlink to 'yolo_dataset_cls_cropped' for convenience."
echo ""
echo "✓ Setup complete! You can now run training scripts."
echo ""

