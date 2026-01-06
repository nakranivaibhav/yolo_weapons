#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

uv run python monte_carlo_influence/monte_carlo_influence.py \
  --model yolo11n.pt \
  --runs 20 \
  --subset-size 3000 \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --conf 0.35 \
  --iou 0.45 \
  --out-dir monte_carlo_influence/runs \
  "$@"
