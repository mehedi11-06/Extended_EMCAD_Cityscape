"""Evaluate a trained EMCAD checkpoint on the Cityscapes validation split.

Writes:
  results/per_class_iou.csv   trainId, class_name, iou, tp, fp, fn
  results/summary.json        mIoU, pixel accuracy, group mIoU, environment
  results/confusion_matrix.npy

Prediction convention: argmax over P[-1], the finest head. This matches the
official EMCAD inference path (utils/utils.py:268 in SLDGroup/EMCAD, which
evaluates `outputs = P[-1]`).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.networks import EMCADNet
from datasets.cityscapes import CityscapesSegmentation, find_cityscapes_root
from utils.metrics import ConfusionMatrix
from utils.cityscapes_labels import NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES
from utils import profiling


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = EMCADNet(
        num_classes=NUM_CLASSES,
        kernel_sizes=cfg.get("kernel_sizes", [1, 3, 5]),
        expansion_factor=cfg.get("expansion_factor", 2),
        dw_parallel=not cfg.get("no_dw_parallel", False),
        add=not cfg.get("concatenation", False),
        lgag_ks=cfg.get("lgag_ks", 3),
        activation=cfg.get("activation_mscb", "relu6"),
        encoder=cfg.get("encoder", "pvt_v2_b2"),
        pretrain=False,                      # weights come from the checkpoint
    )
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[warn] missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    return model.to(device).eval(), ck, cfg


@torch.no_grad()
def evaluate(model, loader, device, amp=True, save_pred_dir=None):
    cm = ConfusionMatrix(NUM_CLASSES, IGNORE_INDEX)
    autocast = torch.amp.autocast("cuda", enabled=amp and device.type == "cuda")
    t0 = time.time()
    for i, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"]
        with autocast:
            P = model(x)
        logits = P[-1] if isinstance(P, (list, tuple)) else P
        pred = logits.float().argmax(1).cpu()
        cm.update(pred, y)
        if save_pred_dir:
            os.makedirs(save_pred_dir, exist_ok=True)
            from PIL import Image
            Image.fromarray(pred[0].numpy().astype(np.uint8)).save(
                os.path.join(save_pred_dir, batch["name"][0] + "_pred_trainIds.png"))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(loader)} images", flush=True)
    return cm, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--val-resize", nargs=2, type=int, default=None,
                    help="H W; omit for native 1024x2048")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-preds", default=None,
                    help="directory for trainId PNGs (for the official-eval cross-check)")
    args = ap.parse_args()

    root = args.data_root or find_cityscapes_root()
    assert root, "could not locate Cityscapes; pass --data-root"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ck, cfg = load_model(args.checkpoint, device)
    val_set = CityscapesSegmentation(root, "val", crop_size=(512, 1024),
                                     val_resize=args.val_resize, limit=args.limit)
    loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2,
                        pin_memory=True)
    print(f"evaluating {len(val_set)} images, "
          f"resize={args.val_resize or 'native'}, masks={val_set.mask_kind}")

    profiling.reset_peak_memory()
    cm, elapsed = evaluate(model, loader, device, amp=not args.no_amp,
                           save_pred_dir=args.save_preds)
    mem_a, mem_r = profiling.peak_memory_gb()

    os.makedirs(args.out_dir, exist_ok=True)
    cm.to_csv(os.path.join(args.out_dir, "per_class_iou.csv"))
    cm.save_matrix(os.path.join(args.out_dir, "confusion_matrix.npy"))

    s = cm.summary()
    s.update({
        "checkpoint": os.path.abspath(args.checkpoint),
        "train_iteration": ck.get("iteration"),
        "n_val_images": len(val_set),
        "val_resize": args.val_resize,
        "eval_amp": not args.no_amp,
        "eval_seconds": elapsed,
        "eval_peak_memory_allocated_gb": mem_a,
        "eval_peak_memory_reserved_gb": mem_r,
        "prediction_head": "P[-1] (finest, matches official EMCAD inference)",
        "train_config": cfg,
        **profiling.env_info(),
    })
    profiling.save_json(s, os.path.join(args.out_dir, "summary.json"))

    print(f"\n{'trainId':>7}  {'class':<14} {'IoU':>7}")
    for i, n in enumerate(CLASS_NAMES):
        v = s["per_class_iou"][n]
        print(f"{i:>7}  {n:<14} {'n/a' if v is None else f'{100*v:6.2f}'}")
    print(f"\nmIoU {100*s['miou']:.2f}  |  pixel acc {100*s['pixel_accuracy']:.2f}  "
          f"|  classes counted {s['n_valid_classes']}/19")
    print("group mIoU: " + "  ".join(f"{k} {100*v:.2f}" for k, v in s["group_miou"].items()))


if __name__ == "__main__":
    main()
