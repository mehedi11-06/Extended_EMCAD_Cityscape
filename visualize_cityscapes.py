"""Qualitative panels: RGB | Ground truth | Prediction | Overlay | Error map.

The error map encodes, per pixel:
    grey   ignored (GT == 255), excluded from metrics
    white  correct
    red    wrong, and the predicted class is in a user-selected focus set
           (default: the thin/small classes the assignment asks about)
    blue   wrong, GT class is in the focus set but the model predicted
           something else  -> a miss of a thin class
    orange wrong, neither side is in the focus set (generic confusion)

So red = false positives for thin classes, blue = false negatives for thin
classes, which is the distinction the failure analysis needs.

Predictions are produced by exactly the same code path as eval_cityscapes.py
(argmax of P[-1], same resize), so panels and metrics cannot disagree.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.cityscapes import CityscapesSegmentation, find_cityscapes_root, denormalize
from eval_cityscapes import load_model
from utils.cityscapes_labels import (
    CLASS_NAMES, CLASS_GROUPS, IGNORE_INDEX, colorize)
from utils.metrics import ConfusionMatrix

FOCUS_DEFAULT = CLASS_GROUPS["small_thin"]


def error_map(pred, gt, focus):
    h, w = gt.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ignored = gt == IGNORE_INDEX
    correct = (pred == gt) & ~ignored
    wrong = (~correct) & ~ignored
    focus = np.asarray(focus)
    pred_focus = np.isin(pred, focus)
    gt_focus = np.isin(gt, focus)

    out[ignored] = (128, 128, 128)
    out[correct] = (255, 255, 255)
    out[wrong & ~pred_focus & ~gt_focus] = (255, 160, 60)   # generic confusion
    out[wrong & pred_focus] = (220, 30, 30)                 # thin-class FP
    out[wrong & gt_focus] = (40, 80, 220)                   # thin-class FN
    return out


def per_image_stats(pred, gt, focus):
    cm = ConfusionMatrix()
    cm.update(pred, gt)
    iou = cm.iou()
    present = [i for i in range(19) if not np.isnan(iou[i])]
    return {
        "miou_image": float(np.nanmean(iou)),
        "classes_present": [CLASS_NAMES[i] for i in present],
        "focus_class_iou": {CLASS_NAMES[i]: (None if np.isnan(iou[i]) else round(float(iou[i]), 4))
                            for i in focus if i in present},
    }


def panel(name, rgb, gt, pred, focus, out_path, dpi=130):
    gt_rgb = colorize(gt)
    pr_rgb = colorize(pred)
    overlay = (0.55 * rgb + 0.45 * pr_rgb).astype(np.uint8)
    err = error_map(pred, gt, focus)

    fig, axes = plt.subplots(1, 5, figsize=(26, 3.4))
    for ax, im, t in zip(axes,
                         [rgb, gt_rgb, pr_rgb, overlay, err],
                         ["Input RGB", "Ground truth", "Prediction",
                          "Prediction overlay", "Error map"]):
        ax.imshow(im)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(name, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--val-resize", nargs=2, type=int, default=[512, 1024])
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--select", nargs="*", default=None,
                    help="explicit image names, e.g. frankfurt_000000_000294")
    ap.add_argument("--strategy", default="spread",
                    choices=["spread", "worst", "best", "most_thin_classes"],
                    help="how to choose scenes when --select is not given")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    root = args.data_root or find_cityscapes_root()
    assert root, "could not locate Cityscapes; pass --data-root"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck, cfg = load_model(args.checkpoint, device)

    val = CityscapesSegmentation(root, "val", crop_size=(512, 1024),
                                 val_resize=args.val_resize)
    loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=2)
    os.makedirs(args.out_dir, exist_ok=True)
    focus = FOCUS_DEFAULT

    records = []
    autocast = torch.amp.autocast("cuda", enabled=not args.no_amp and device.type == "cuda")
    with torch.no_grad():
        for batch in loader:
            name = batch["name"][0]
            if args.select and name not in args.select:
                continue
            x = batch["image"].to(device)
            with autocast:
                P = model(x)
            logits = P[-1] if isinstance(P, (list, tuple)) else P
            pred = logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
            gt = batch["label"][0].numpy().astype(np.uint8)
            st = per_image_stats(pred, gt, focus)
            st["name"] = name
            st["n_thin_classes"] = len(st["focus_class_iou"])
            records.append((st, batch["image"][0].clone(), gt, pred))
            if args.select and len(records) >= len(args.select):
                break

    if not args.select:
        if args.strategy == "worst":
            records.sort(key=lambda r: r[0]["miou_image"])
        elif args.strategy == "best":
            records.sort(key=lambda r: -r[0]["miou_image"])
        elif args.strategy == "most_thin_classes":
            records.sort(key=lambda r: -r[0]["n_thin_classes"])
        else:
            # "spread": rank by image mIoU and sample evenly across the range,
            # so the figures are representative rather than only the best cases
            records.sort(key=lambda r: r[0]["miou_image"])
            idx = np.linspace(0, len(records) - 1, args.num_images).astype(int)
            records = [records[i] for i in idx]
        records = records[: args.num_images]

    manifest = []
    for st, img, gt, pred in records:
        rgb = denormalize(img)
        out_path = os.path.join(args.out_dir, f"qualitative_{st['name']}.png")
        panel(st["name"], rgb, gt, pred, focus, out_path)
        st["figure"] = out_path
        manifest.append(st)
        print(f"wrote {out_path}  image mIoU {st['miou_image']:.3f}")

    with open(os.path.join(args.out_dir, "qualitative_manifest.json"), "w") as f:
        json.dump({"selection_strategy": args.strategy,
                   "focus_classes": [CLASS_NAMES[i] for i in focus],
                   "val_resize": args.val_resize,
                   "checkpoint": os.path.abspath(args.checkpoint),
                   "images": manifest}, f, indent=2)
    print(f"\nmanifest: {os.path.join(args.out_dir, 'qualitative_manifest.json')}")


if __name__ == "__main__":
    main()
