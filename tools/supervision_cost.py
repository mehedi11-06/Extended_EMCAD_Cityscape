"""Measure the real cost of EMCAD's three supervision modes at Cityscapes size.

Turns the supervision choice from an opinion into a measurement. Runs a few
real training steps per mode (synthetic tensors, same shapes as training) and
records peak memory and step time, then writes results/supervision_cost.csv.

    python -m tools.supervision_cost --crop 512 1024 --batch 2 --amp
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.networks import EMCADNet
from losses.segmentation_losses import CombinedSegLoss, supervision_subsets, supervised_loss
from utils.cityscapes_labels import NUM_CLASSES, IGNORE_INDEX
from utils import profiling


def run_mode(mode, crop, batch, amp, iters, warmup, device, pretrained_dir, pretrain):
    H, W = crop
    model = EMCADNet(num_classes=NUM_CLASSES, pretrain=pretrain,
                     pretrained_dir=pretrained_dir).to(device).train()
    crit = CombinedSegLoss(NUM_CLASSES, IGNORE_INDEX, 0.3, 0.7).to(device)
    subsets = supervision_subsets(mode, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    x = torch.randn(batch, 3, H, W, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch, H, W), device=device)
    y[:, : H // 8, :] = IGNORE_INDEX

    profiling.reset_peak_memory()
    timer = profiling.StepTimer(warmup=warmup)
    oom = False
    try:
        for _ in range(iters + warmup):
            timer.start()
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                P = model(x, mode="train")
                loss, _ = supervised_loss(P, y, crit, subsets)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            timer.stop()
    except torch.cuda.OutOfMemoryError:
        oom = True
    mem_a, mem_r = profiling.peak_memory_gb()
    del model, opt, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"supervision": mode, "n_loss_terms": len(subsets),
            "oom": oom,
            "peak_alloc_gb": round(mem_a, 3), "peak_reserved_gb": round(mem_r, 3),
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in timer.report().items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", nargs=2, type=int, default=[512, 1024])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pretrained-dir", default="./pretrained_pth/pvt/")
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--out", default="results/supervision_cost.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for mode in ["last_layer", "deep_supervision", "mutation"]:
        print(f"-- {mode}")
        r = run_mode(mode, args.crop, args.batch, args.amp, args.iters,
                     args.warmup, device, args.pretrained_dir, not args.no_pretrain)
        r.update({"crop": f"{args.crop[0]}x{args.crop[1]}", "batch": args.batch,
                  "amp": args.amp, "gpu": profiling.env_info()["gpu_name"]})
        print("   ", r)
        rows.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    base = next((r for r in rows if r["supervision"] == "last_layer"), None)
    if base and base["mean_step_time_s"] == base["mean_step_time_s"]:
        for r in rows:
            if not r["oom"]:
                print(f"  {r['supervision']:<18} {r['n_loss_terms']:>2} terms  "
                      f"{r['mean_step_time_s']:.3f}s "
                      f"({r['mean_step_time_s']/base['mean_step_time_s']:.2f}x)  "
                      f"{r['peak_alloc_gb']:.2f} GB "
                      f"({r['peak_alloc_gb']/base['peak_alloc_gb']:.2f}x)")
            else:
                print(f"  {r['supervision']:<18} OUT OF MEMORY at this configuration")


if __name__ == "__main__":
    main()
