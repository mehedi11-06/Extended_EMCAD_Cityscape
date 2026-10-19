"""Train EMCAD on Cityscapes, 19 classes.

Derived from the official EMCAD train_synapse.py + trainer.py (SLDGroup/EMCAD,
CVPR 2024). The model in lib/ is used unmodified; this script replaces the
Synapse task pipeline with a Cityscapes one.

Differences from the original trainer, all deliberate and documented:
  * CrossEntropyLoss(ignore_index=255)            (original has no ignore_index)
  * masked multiclass Dice                        (original Dice counts ignored pixels)
  * iteration-based schedule + poly LR + AMP      (fits a bounded Kaggle session)
  * confusion-matrix mIoU validation              (original uses medical Dice/HD95)
  * resumable checkpoints + wall-clock budget     (sessions have a hard limit)

Usage:
  python train_cityscapes.py --config configs/cityscapes_pvtv2b2.yaml
  python train_cityscapes.py --config ... --resume runs/<exp>/last.pth
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.networks import EMCADNet                       # unmodified official model
from datasets.cityscapes import CityscapesSegmentation, find_cityscapes_root
from losses.segmentation_losses import (
    CombinedSegLoss, supervision_subsets, supervised_loss)
from utils.metrics import ConfusionMatrix
from utils.cityscapes_labels import NUM_CLASSES, IGNORE_INDEX
from utils import profiling


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    data_root=None, out_dir="runs/emcad_cityscapes", encoder="pvt_v2_b2",
    pretrained_dir="./pretrained_pth/pvt/", no_pretrain=False,
    kernel_sizes=[1, 3, 5], expansion_factor=2, lgag_ks=3,
    activation_mscb="relu6", no_dw_parallel=False, concatenation=False,
    supervision="deep_supervision", loss="ce_dice", w_ce=0.3, w_dice=0.7,
    w_focal=0.0, focal_gamma=2.0,
    crop_size=[512, 1024], val_resize=[512, 1024], scale_min=0.5, scale_max=2.0,
    batch_size=2, num_workers=2, lr=6e-5, weight_decay=0.01, poly_power=0.9,
    warmup_iters=500, max_iters=20000, val_interval=2000, log_interval=50,
    amp=True, seed=1234, deterministic=True, max_hours=0.0,
    train_limit=None, val_limit=None, grad_accum=1,
)


def load_config(path, overrides):
    cfg = dict(DEFAULTS)
    if path:
        import yaml
        with open(path) as f:
            cfg.update({k: v for k, v in (yaml.safe_load(f) or {}).items()})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def build_model(cfg, device):
    model = EMCADNet(
        num_classes=NUM_CLASSES,                 # <-- the entire architectural change
        kernel_sizes=cfg["kernel_sizes"],
        expansion_factor=cfg["expansion_factor"],
        dw_parallel=not cfg["no_dw_parallel"],
        add=not cfg["concatenation"],
        lgag_ks=cfg["lgag_ks"],
        activation=cfg["activation_mscb"],
        encoder=cfg["encoder"],
        pretrain=not cfg["no_pretrain"],
        pretrained_dir=cfg["pretrained_dir"],
    )
    return model.to(device)


def poly_lr(base_lr, it, max_iters, power=0.9, warmup=0):
    if warmup and it < warmup:
        return base_lr * (it + 1) / warmup
    t = (it - warmup) / max(max_iters - warmup, 1)
    return base_lr * (1 - min(t, 1.0)) ** power


@torch.no_grad()
def validate(model, loader, device, amp=True, max_batches=None):
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES, IGNORE_INDEX)
    autocast = torch.amp.autocast("cuda", enabled=amp and device.type == "cuda")
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"]
        with autocast:
            P = model(x)
        logits = P[-1] if isinstance(P, (list, tuple)) else P   # official convention
        pred = logits.float().argmax(1).cpu()
        cm.update(pred, y)
    model.train()
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", dest=k,
                            action="store_true", default=None)
        elif isinstance(v, list):
            ap.add_argument(f"--{k.replace('_', '-')}", dest=k, nargs="+",
                            type=type(v[0]), default=None)
        else:
            t = type(v) if v is not None else str
            ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=t, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, {k: getattr(args, k) for k in DEFAULTS})
    if cfg["data_root"] is None:
        cfg["data_root"] = find_cityscapes_root()
    assert cfg["data_root"], "could not locate Cityscapes; pass --data-root"

    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg["seed"], cfg["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(cfg["out_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ---------------- data ----------------
    train_set = CityscapesSegmentation(
        cfg["data_root"], "train", crop_size=cfg["crop_size"],
        scale_range=(cfg["scale_min"], cfg["scale_max"]),
        limit=cfg["train_limit"], seed=cfg["seed"])
    val_set = CityscapesSegmentation(
        cfg["data_root"], "val", crop_size=cfg["crop_size"],
        val_resize=cfg["val_resize"], limit=cfg["val_limit"], seed=cfg["seed"])
    print(f"train {len(train_set)} | val {len(val_set)} | masks from {train_set.mask_kind}")

    def worker_init_fn(wid):
        random.seed(cfg["seed"] + wid)
        np.random.seed(cfg["seed"] + wid)

    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn, generator=g,
        persistent_workers=cfg["num_workers"] > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)

    # ---------------- model / loss / optim ----------------
    model = build_model(cfg, device)
    subsets = supervision_subsets(cfg["supervision"], 4)
    print(f"supervision={cfg['supervision']} -> {len(subsets)} loss terms: {subsets}")

    w_dice = cfg["w_dice"] if cfg["loss"] in ("ce_dice", "ce_dice_focal") else 0.0
    w_focal = cfg["w_focal"] if cfg["loss"] in ("ce_focal", "ce_dice_focal") else 0.0
    w_ce = 1.0 if cfg["loss"] == "ce" else cfg["w_ce"]
    criterion = CombinedSegLoss(NUM_CLASSES, IGNORE_INDEX, w_ce, w_dice,
                                w_focal, cfg["focal_gamma"]).to(device)
    print(f"loss={cfg['loss']} (w_ce={w_ce} w_dice={w_dice} w_focal={w_focal})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_iter, best_miou = 0, -1.0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scaler") and use_amp:
            scaler.load_state_dict(ck["scaler"])
        start_iter = ck.get("iteration", 0)
        best_miou = ck.get("best_miou", -1.0)
        print(f"resumed from {args.resume} at iter {start_iter} (best mIoU {best_miou:.4f})")

    # ---------------- logging ----------------
    csv_path = os.path.join(cfg["out_dir"], "train_log.csv")
    new_csv = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if new_csv:
        csv_w.writerow(["iter", "epoch_frac", "total_loss", "ce", "dice", "focal",
                        "lr", "step_time_s", "data_wait_s", "peak_mem_gb",
                        "val_miou", "best_miou"])

    timer = profiling.StepTimer(warmup=20)
    profiling.reset_peak_memory()
    t_start = time.time()
    it = start_iter
    data_wait = 0.0
    model.train()
    stop = False

    print(f"training {cfg['max_iters']} iters on {device} "
          f"(amp={use_amp}, batch={cfg['batch_size']}, crop={cfg['crop_size']})")

    while not stop:
        t_data = time.time()
        for batch in train_loader:
            data_wait = time.time() - t_data
            if it >= cfg["max_iters"]:
                stop = True
                break

            lr = poly_lr(cfg["lr"], it, cfg["max_iters"],
                         cfg["poly_power"], cfg["warmup_iters"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            timer.start()
            with torch.amp.autocast("cuda", enabled=use_amp):
                P = model(x, mode="train")
                loss, parts = supervised_loss(P, y, criterion, subsets)
                loss = loss / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if (it + 1) % cfg["grad_accum"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            step_time = timer.stop()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at iter {it}: {loss.item()}. "
                    "Check label range and AMP; see MANUAL.md section 27.")

            if it % cfg["log_interval"] == 0:
                mem_a, _ = profiling.peak_memory_gb()
                print(f"iter {it:6d}/{cfg['max_iters']} loss {loss.item()*cfg['grad_accum']:.4f} "
                      f"ce {parts.get('ce', 0):.4f} dice {parts.get('dice', 0):.4f} "
                      f"lr {lr:.2e} step {step_time:.3f}s data {data_wait:.3f}s "
                      f"mem {mem_a:.2f}GB", flush=True)
                csv_w.writerow([it, round(it * cfg["batch_size"] / len(train_set), 3),
                                f"{loss.item()*cfg['grad_accum']:.6f}",
                                f"{parts.get('ce', float('nan')):.6f}",
                                f"{parts.get('dice', float('nan')):.6f}",
                                f"{parts.get('focal', float('nan')):.6f}",
                                f"{lr:.3e}", f"{step_time:.4f}", f"{data_wait:.4f}",
                                f"{mem_a:.3f}", "", f"{best_miou:.6f}"])
                csv_f.flush()

            it += 1

            due_val = it % cfg["val_interval"] == 0 or it >= cfg["max_iters"]
            out_of_time = (cfg["max_hours"] > 0
                           and (time.time() - t_start) / 3600 > cfg["max_hours"])
            if due_val or out_of_time:
                cm = validate(model, val_loader, device, amp=use_amp)
                miou = cm.miou()
                mem_a, mem_r = profiling.peak_memory_gb()
                print(f"[val] iter {it} mIoU {miou:.4f} "
                      f"(best {max(best_miou, miou):.4f}) peak_alloc {mem_a:.2f}GB", flush=True)
                csv_w.writerow([it, "", "", "", "", "", "", "", "",
                                f"{mem_a:.3f}", f"{miou:.6f}",
                                f"{max(best_miou, miou):.6f}"])
                csv_f.flush()

                ck = {"model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict() if use_amp else None,
                      "iteration": it, "best_miou": max(best_miou, miou),
                      "val_miou": miou, "config": cfg,
                      "env": profiling.env_info()}
                torch.save(ck, os.path.join(cfg["out_dir"], "last.pth"))
                if miou > best_miou:
                    best_miou = miou
                    torch.save(ck, os.path.join(cfg["out_dir"], "best_miou.pth"))
                    cm.to_csv(os.path.join(cfg["out_dir"], "per_class_iou_val.csv"))

                if out_of_time:
                    print(f"wall-clock budget {cfg['max_hours']}h reached; "
                          f"checkpoint saved at iter {it}. Resume with "
                          f"--resume {cfg['out_dir']}/last.pth")
                    stop = True
                    break
            t_data = time.time()

    mem_a, mem_r = profiling.peak_memory_gb()
    summary = {
        "iterations_completed": it,
        "best_val_miou": best_miou,
        "peak_memory_allocated_gb": mem_a,
        "peak_memory_reserved_gb": mem_r,
        "amp": use_amp,
        "batch_size": cfg["batch_size"],
        "crop_size": cfg["crop_size"],
        "supervision": cfg["supervision"],
        "loss": cfg["loss"],
        "n_loss_terms": len(subsets),
        "wall_clock_hours": (time.time() - t_start) / 3600,
        **timer.report(),
        **profiling.env_info(),
        "config": cfg,
    }
    profiling.save_json(summary, os.path.join(cfg["out_dir"], "train_summary.json"))
    csv_f.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=str))


if __name__ == "__main__":
    main()
