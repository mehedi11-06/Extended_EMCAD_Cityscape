"""Generate *_gtFine_labelTrainIds.png from *_gtFine_labelIds.png.

Only needed when the Cityscapes mirror does not already ship trainIds. Prefer
the official tool when internet is available:

    pip install cityscapesscripts
    CITYSCAPES_DATASET=/path/to/cityscapes csCreateTrainIdLabelImgs

This script applies the identical mapping table (utils/cityscapes_labels.py,
cross-checked against cityscapesscripts when installed) and writes into
--out-root so the read-only /kaggle/input mount is never touched.

    python -m tools.make_train_ids --data-root /kaggle/input/<ds>/cityscapes \
        --out-root /kaggle/working/cityscapes_trainids
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cityscapes_labels import build_id_to_trainid_lut, verify_against_cityscapesscripts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--link-images", action="store_true",
                    help="symlink leftImg8bit into out-root so it is a complete tree")
    args = ap.parse_args()

    ok, msg = verify_against_cityscapesscripts()
    print(msg)
    if ok is False:
        sys.exit("refusing to remap with a table that disagrees with the official one")

    lut = build_id_to_trainid_lut()
    total = 0
    for split in args.splits:
        pattern = os.path.join(args.data_root, "gtFine", split, "*", "*_gtFine_labelIds.png")
        files = sorted(glob.glob(pattern))
        print(f"{split}: {len(files)} labelIds files")
        for p in files:
            rel = os.path.relpath(p, args.data_root)
            out = os.path.join(args.out_root, rel).replace("_labelIds.png", "_labelTrainIds.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            m = np.array(Image.open(p))
            Image.fromarray(lut[m]).save(out)
            total += 1
        if args.link_images:
            src = os.path.join(args.data_root, "leftImg8bit", split)
            dst = os.path.join(args.out_root, "leftImg8bit", split)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copytree(src, dst)
    print(f"wrote {total} trainId masks to {args.out_root}")
    print("verify with: python -m tests.test_dataset --data-root " + args.out_root)


if __name__ == "__main__":
    main()
