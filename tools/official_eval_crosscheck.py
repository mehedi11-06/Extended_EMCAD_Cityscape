"""Cross-check the custom evaluator against the official Cityscapes scripts.

The official evaluator (csEvaluationPbs / evalPixelLevelSemanticLabeling)
expects prediction PNGs in the **labelId** space (0..33), not trainIds, and
matches them to ground truth by filename. This script:

  1. converts the trainId prediction PNGs written by
     `eval_cityscapes.py --save-preds` into labelId PNGs,
  2. runs the official evaluator if cityscapesscripts is installed,
  3. prints the official mIoU next to the custom one so they can be compared.

Agreement to ~1e-4 means the custom confusion matrix is correct. A systematic
gap usually means a resize mismatch: the official evaluator compares at native
1024x2048, so run eval_cityscapes.py WITHOUT --val-resize for this check.

    python eval_cityscapes.py --checkpoint runs/<exp>/best_miou.pth \
        --save-preds results/preds_trainids --out-dir results/native
    python -m tools.official_eval_crosscheck \
        --pred-dir results/preds_trainids --data-root <root> \
        --custom-summary results/native/summary.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cityscapes_labels import LABELS, IGNORE_INDEX


def trainid_to_labelid_lut():
    lut = np.zeros(256, dtype=np.uint8)
    for l in LABELS:
        if l.trainId not in (255, -1) and l.id >= 0:
            lut[l.trainId] = l.id
    lut[IGNORE_INDEX] = 0            # unlabeled
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="trainId PNGs from eval_cityscapes.py")
    ap.add_argument("--out-dir", default="results/preds_labelids")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--custom-summary", default=None)
    ap.add_argument("--run-official", action="store_true")
    args = ap.parse_args()

    lut = trainid_to_labelid_lut()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.pred_dir, "*_pred_trainIds.png")))
    assert files, f"no predictions in {args.pred_dir}"
    for p in files:
        m = np.array(Image.open(p))
        name = os.path.basename(p).replace("_pred_trainIds.png", "")
        Image.fromarray(lut[m]).save(os.path.join(args.out_dir, f"{name}*.png".replace("*", "")))
    print(f"converted {len(files)} predictions to labelId space in {args.out_dir}")
    print("NOTE: the official evaluator compares at native 1024x2048. If the "
          "predictions were produced at a reduced resolution they must be upsampled "
          "with NEAREST to native size before this comparison is meaningful.")

    if args.run_official:
        env = dict(os.environ, CITYSCAPES_DATASET=args.data_root,
                   CITYSCAPES_RESULTS=args.out_dir)
        try:
            out = subprocess.run(
                [sys.executable, "-m",
                 "cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling"],
                env=env, capture_output=True, text=True, timeout=3600)
            print(out.stdout[-4000:])
            if out.returncode != 0:
                print(out.stderr[-2000:])
        except FileNotFoundError:
            print("cityscapesscripts not installed: pip install cityscapesscripts")

    if args.custom_summary and os.path.exists(args.custom_summary):
        s = json.load(open(args.custom_summary))
        print(f"\ncustom evaluator mIoU: {100*s['miou']:.4f} "
              f"over {s['n_valid_classes']} classes")
        print("Compare this against the official 'average' IoU class figure above. "
              "Differences beyond ~0.01 point to a resize, ignore-handling or "
              "class-ordering discrepancy, in that order of likelihood.")


if __name__ == "__main__":
    main()
