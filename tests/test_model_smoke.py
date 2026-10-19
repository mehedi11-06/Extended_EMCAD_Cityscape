"""Milestone 1 gate: EMCADNet at num_classes=19 forward / loss / backward / step.

Run (from the repo root, with lib/ present):
    python -m tests.test_model_smoke --crop 512 1024 --batch 2
    python -m tests.test_model_smoke --crop 128 256 --batch 1 --cpu   # quick
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.networks import EMCADNet
from losses.segmentation_losses import (
    CombinedSegLoss, supervision_subsets, supervised_loss)
from utils.cityscapes_labels import NUM_CLASSES, IGNORE_INDEX
from utils import profiling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", nargs=2, type=int, default=[512, 1024])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--encoder", default="pvt_v2_b2")
    ap.add_argument("--pretrained-dir", default="./pretrained_pth/pvt/")
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--supervision", default="deep_supervision")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--check-divisibility", action="store_true",
                    help="also prove a non-divisible input fails loudly")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    H, W = args.crop
    assert H % 32 == 0 and W % 32 == 0, "crop must be divisible by 32"

    print(f"device={device}  crop={H}x{W}  batch={args.batch}  "
          f"supervision={args.supervision}  amp={args.amp}")

    model = EMCADNet(num_classes=NUM_CLASSES, encoder=args.encoder,
                     pretrain=not args.no_pretrain,
                     pretrained_dir=args.pretrained_dir).to(device).train()
    n_enc = sum(p.numel() for p in model.backbone.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_head = sum(p.numel() for n, p in model.named_parameters() if "out_head" in n)
    print(f"params: encoder {n_enc:,}  decoder {n_dec:,}  heads {n_head:,}  "
          f"total {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(args.batch, 3, H, W, device=device)
    y = torch.randint(0, NUM_CLASSES, (args.batch, H, W), device=device)
    y[:, : H // 8, :] = IGNORE_INDEX          # simulate an ignore region

    profiling.reset_peak_memory()
    with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
        P = model(x, mode="train")

    assert isinstance(P, (list, tuple)) and len(P) == 4, type(P)
    for i, p in enumerate(P):
        print(f"  P[{i}] shape={tuple(p.shape)} dtype={p.dtype} "
              f"min={p.min().item():.3f} max={p.max().item():.3f}")
        assert tuple(p.shape) == (args.batch, NUM_CLASSES, H, W), tuple(p.shape)
        assert torch.isfinite(p).all(), f"P[{i}] contains non-finite values"
    assert not torch.allclose(P[0], P[1]), "heads produce identical output -- suspicious"

    # CE only, so the expected value at init is analytically known
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    ce_sum = sum(ce(p.float(), y.long()) for p in P)
    expected = 4 * math.log(NUM_CLASSES)
    print(f"  sum of 4 CE terms = {ce_sum.item():.4f} "
          f"(expected near 4*ln(19) = {expected:.4f} at init)")
    assert torch.isfinite(ce_sum)
    assert abs(ce_sum.item() - expected) < 0.35 * expected, (
        "CE far from its initialization value -- check label range / head init")

    crit = CombinedSegLoss(NUM_CLASSES, IGNORE_INDEX, w_ce=0.3, w_dice=0.7).to(device)
    subsets = supervision_subsets(args.supervision, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
        P = model(x, mode="train")
        loss, parts = supervised_loss(P, y, crit, subsets)
    scaler.scale(loss).backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)

    mem_a, mem_r = profiling.peak_memory_gb()
    print(f"  combined loss {loss.item():.4f} parts={ {k: round(v,4) for k,v in parts.items()} }")
    print(f"  grad norm {float(gnorm):.3f}  ({len(subsets)} loss terms)")
    print(f"  peak allocated {mem_a:.2f} GB   reserved {mem_r:.2f} GB")
    assert torch.isfinite(loss) and math.isfinite(float(gnorm)) and float(gnorm) > 0

    if args.check_divisibility:
        try:
            model(torch.randn(1, 3, H + 1, W + 1, device=device))
            print("  [unexpected] non-divisible input did NOT raise")
        except RuntimeError as e:
            print(f"  non-divisible input raises RuntimeError (good, fails loudly): "
                  f"{str(e)[:90]}")

    print("MODEL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
