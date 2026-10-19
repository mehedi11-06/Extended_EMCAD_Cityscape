"""Milestone 4 gate: overfit a tiny fixed subset of Cityscapes.

The point is not accuracy, it is proving the whole chain -- loading, label
mapping, augmentation (off here), model, loss masking, backward, optimizer,
metric -- is wired correctly. If a model cannot memorize 12 images it will
never learn 2975, and a long run would only hide the bug for hours.

Design:
  * N images (default 12), sampled with a FIXED seed so the subset is
    reproducible and reportable.
  * Augmentation OFF (deterministic centre crop). Augmentation is a
    regularizer; leaving it on fights memorization and muddies the signal.
  * Train and evaluate on the SAME images -- this is a correctness probe,
    not a generalization measurement. Say so in the README.
  * The pass criterion is computed only over classes actually PRESENT in the
    subset. A 12-image subset will not contain all 19 classes, so a flat
    "mIoU > 0.9" over 19 classes would be unreachable by construction.

    python -m tools.tiny_overfit --data-root <root> --images 12 --iters 300
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.networks import EMCADNet
from datasets.cityscapes import CityscapesSegmentation, find_cityscapes_root
from losses.segmentation_losses import CombinedSegLoss, supervision_subsets, supervised_loss
from utils.metrics import ConfusionMatrix
from utils.cityscapes_labels import NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES
from utils import profiling


@torch.no_grad()
def evaluate_subset(model, loader, device, amp):
    model.eval()
    cm = ConfusionMatrix()
    for b in loader:
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            P = model(b["image"].to(device))
        logits = P[-1] if isinstance(P, (list, tuple)) else P
        cm.update(logits.float().argmax(1).cpu(), b["label"])
    model.train()
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--crop", nargs=2, type=int, default=[512, 1024])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--supervision", default="deep_supervision")
    ap.add_argument("--loss", default="ce_dice", choices=["ce", "ce_dice"])
    ap.add_argument("--pretrained-dir", default="./pretrained_pth/pvt/")
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="logs/overfit.json")
    ap.add_argument("--min-pixels", type=int, default=2000,
                    help="a class counts toward the criterion only if the subset "
                         "contains at least this many of its pixels")
    args = ap.parse_args()

    root = args.data_root or find_cityscapes_root()
    assert root, "could not locate Cityscapes; pass --data-root"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CityscapesSegmentation(root, "train", crop_size=args.crop,
                                augment=False, limit=args.images, seed=args.seed)
    print(f"tiny-overfit subset ({len(ds)} images, augmentation OFF):")
    for i in range(len(ds)):
        print("   ", ds.name(i))

    # which classes are actually present, and with how many pixels
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(len(ds)):
        y = ds[i]["label"].numpy()
        counts += np.bincount(y[y != IGNORE_INDEX], minlength=NUM_CLASSES)[:NUM_CLASSES]
    present = [i for i in range(NUM_CLASSES) if counts[i] >= args.min_pixels]
    print(f"\nclasses with >= {args.min_pixels} px in the subset: {len(present)}/19")
    for i in present:
        print(f"    {i:2d} {CLASS_NAMES[i]:<14} {counts[i]:>9,} px")
    rare = [i for i in range(NUM_CLASSES) if 0 < counts[i] < args.min_pixels]
    if rare:
        print("  present but too rare to score: "
              + ", ".join(f"{CLASS_NAMES[i]}({counts[i]})" for i in rare))

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=False)
    eval_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    model = EMCADNet(num_classes=NUM_CLASSES, pretrain=not args.no_pretrain,
                     pretrained_dir=args.pretrained_dir).to(device).train()
    crit = CombinedSegLoss(NUM_CLASSES, IGNORE_INDEX,
                           w_ce=1.0 if args.loss == "ce" else 0.3,
                           w_dice=0.0 if args.loss == "ce" else 0.7).to(device)
    subsets = supervision_subsets(args.supervision, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    profiling.reset_peak_memory()
    history, it, t0 = [], 0, time.time()
    first_loss = None
    while it < args.iters:
        for b in loader:
            if it >= args.iters:
                break
            x, y = b["image"].to(device), b["label"].to(device)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                P = model(x, mode="train")
                loss, parts = supervised_loss(P, y, crit, subsets)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if first_loss is None:
                first_loss = float(loss)
            if it % 25 == 0:
                print(f"  iter {it:4d}  loss {float(loss):.4f}  "
                      f"ce {parts.get('ce',0):.4f}  dice {parts.get('dice',0):.4f}", flush=True)
                history.append({"iter": it, "loss": float(loss), **parts})
            it += 1

    cm = evaluate_subset(model, eval_loader, device, args.amp)
    iou = cm.iou()
    scored = [i for i in present if not np.isnan(iou[i])]
    subset_miou = float(np.mean([iou[i] for i in scored])) if scored else float("nan")
    final_loss = float(loss)
    mem_a, _ = profiling.peak_memory_gb()

    print(f"\nper-class IoU on the training subset:")
    for i in range(NUM_CLASSES):
        tag = "scored" if i in scored else ("rare" if counts[i] else "absent")
        v = iou[i]
        print(f"  {i:2d} {CLASS_NAMES[i]:<14} "
              f"{'   n/a' if np.isnan(v) else f'{100*v:6.2f}'}   [{tag}]")

    pass_loss = final_loss < 0.25 * first_loss
    pass_miou = subset_miou > 0.85
    pass_cov = len(scored) >= max(5, int(0.5 * len(present)))
    print(f"\nfirst loss {first_loss:.4f} -> final {final_loss:.4f}")
    print(f"subset mIoU over {len(scored)} scored classes: {100*subset_miou:.2f}")
    print(f"peak allocated {mem_a:.2f} GB, {time.time()-t0:.0f}s for {args.iters} iters")
    print(f"\ncriteria: loss collapsed {pass_loss} | subset mIoU>0.85 {pass_miou} "
          f"| coverage {pass_cov}")

    result = {
        "images": [ds.name(i) for i in range(len(ds))],
        "iters": args.iters, "augmentation": False,
        "supervision": args.supervision, "loss": args.loss,
        "first_loss": first_loss, "final_loss": final_loss,
        "scored_classes": [CLASS_NAMES[i] for i in scored],
        "subset_miou_scored_classes": subset_miou,
        "per_class_iou": {CLASS_NAMES[i]: (None if np.isnan(iou[i]) else float(iou[i]))
                          for i in range(NUM_CLASSES)},
        "class_pixel_counts": {CLASS_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)},
        "passed": bool(pass_loss and pass_miou and pass_cov),
        "peak_memory_allocated_gb": mem_a,
        **profiling.env_info(),
    }
    profiling.save_json(result, args.out)
    print(f"\nwrote {args.out}")
    print("OVERFIT TEST PASSED" if result["passed"] else "OVERFIT TEST FAILED -- do not start full training")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
