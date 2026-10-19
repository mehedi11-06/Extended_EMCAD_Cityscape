#!/usr/bin/env bash
# Primary experiment: EMCAD + PVTv2-B2, 19 classes, CE + masked Dice.
set -euo pipefail
ROOT="${1:?usage: scripts/train.sh /path/to/cityscapes}"
python -W ignore train_cityscapes.py \
  --config configs/cityscapes_pvtv2b2.yaml \
  --data-root "$ROOT"
