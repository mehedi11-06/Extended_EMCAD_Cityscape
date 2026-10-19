#!/usr/bin/env bash
# Final validation + figures + failure analysis from one checkpoint.
set -euo pipefail
ROOT="${1:?usage: scripts/evaluate.sh /path/to/cityscapes [checkpoint]}"
CKPT="${2:-/kaggle/working/runs/emcad_cityscapes_ce_dice/best_miou.pth}"
python -W ignore eval_cityscapes.py --checkpoint "$CKPT" --data-root "$ROOT" \
  --val-resize 512 1024 --out-dir results
python -W ignore visualize_cityscapes.py --checkpoint "$CKPT" --data-root "$ROOT" \
  --num-images 8 --strategy spread --out-dir figures
python -m tools.analyze_failures
