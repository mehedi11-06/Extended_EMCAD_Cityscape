"""Generates kaggle/EMCAD_Cityscapes_Kaggle.ipynb from the cell list below.

Keeping the notebook in a generator keeps it diffable in git. Run:
    python kaggle/build_notebook.py
"""
import json
import os

MD = "markdown"
CODE = "code"

CELLS = [
(MD, """# EMCAD → Cityscapes (19 classes) — Kaggle runner

Each section is one **milestone gate**. Do not run a later gate until the
previous one printed `PASSED`. Full rationale: `MANUAL.md` in the repo.

**Kaggle session settings (right-hand pane):**
- Accelerator: **GPU P100** (or T4 x2 — this code uses one GPU)
- Internet: **On** for the first setup run (pip + git clone), then it can be off
- Persistence: *Variables and Files* if available, so `/kaggle/working` survives a restart
- Attach: your Cityscapes dataset, and a dataset containing `pvt_v2_b2.pth`

Session length and weekly GPU quota are enforced by Kaggle and change over
time — check the pane. Every long step below is **resumable**, so a session
ending is an inconvenience, not a lost run."""),

(CODE, """# M0.1 — setup. Run once per session.
import os, subprocess, sys, textwrap
WORK = "/kaggle/working"
PROJ = f"{WORK}/EMCAD-Cityscapes"

# Option A: your fork/repo containing this project (recommended — keeps history)
REPO_URL = ""           # e.g. "https://github.com/<you>/EMCAD-Cityscapes.git"
# Option B: the project uploaded as a Kaggle dataset
PROJ_DATASET = ""       # e.g. "/kaggle/input/emcad-cityscapes-code"

if REPO_URL:
    if not os.path.isdir(PROJ):
        subprocess.run(["git", "clone", REPO_URL, PROJ], check=True)
elif PROJ_DATASET:
    subprocess.run(["cp", "-r", PROJ_DATASET, PROJ], check=True)
else:
    raise SystemExit("set REPO_URL or PROJ_DATASET first")

os.chdir(PROJ)
print(os.getcwd(), os.listdir())"""),

(CODE, """# M0.2 — official EMCAD lib/ (unmodified) + dependencies
import os, subprocess
if not os.path.exists("lib/networks.py"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/SLDGroup/EMCAD.git", "/kaggle/working/EMCAD"], check=True)
    subprocess.run(["cp", "-r", "/kaggle/working/EMCAD/lib", "."], check=True)
subprocess.run(["pip", "install", "-q", "timm", "pyyaml"], check=True)
print(sorted(os.listdir("lib")))"""),

(CODE, """# M0.3 — pretrained PVTv2-B2 into ./pretrained_pth/pvt/
import glob, os, shutil
os.makedirs("pretrained_pth/pvt", exist_ok=True)
hits = glob.glob("/kaggle/input/**/pvt_v2_b2.pth", recursive=True)
print("candidates:", hits)
if hits and not os.path.exists("pretrained_pth/pvt/pvt_v2_b2.pth"):
    shutil.copy(hits[0], "pretrained_pth/pvt/pvt_v2_b2.pth")
print(os.listdir("pretrained_pth/pvt"))
# If empty: attach a dataset with the PVTv2-B2 checkpoint, or train with
# --no-pretrain and state that clearly in the README (it will cost mIoU)."""),

(CODE, """# M0.4 — torch>=2.6 compatibility for lib/networks.py:68 (torch.load default changed)
import re, torch
src = open("lib/networks.py").read()
if tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 6) \\
        and "weights_only" not in src:
    src = src.replace("save_model = torch.load(path)",
                      "save_model = torch.load(path, weights_only=False)")
    open("lib/networks.py", "w").write(src)
    print("patched lib/networks.py:68 -> torch.load(..., weights_only=False)")
else:
    print("no patch needed")"""),

(CODE, """# M0 GATE — environment, dataset discovery, weights
!python -m tools.check_env"""),

(CODE, """# Set the data root once; every later cell uses it.
from datasets.cityscapes import find_cityscapes_root
ROOT = find_cityscapes_root() or "/kaggle/input/<your-cityscapes-dataset>/cityscapes"
print("ROOT =", ROOT)
%env CS_ROOT=$ROOT"""),

(CODE, """# Only if the mirror ships labelIds but not labelTrainIds:
# !python -m tools.make_train_ids --data-root $CS_ROOT \\
#     --out-root /kaggle/working/cityscapes_trainids --link-images
# then set ROOT = "/kaggle/working/cityscapes_trainids" and re-run check_env"""),

(CODE, """# M1 GATE — model forward/loss/backward/step at 19 classes, real crop size
!python -m tests.test_model_smoke --crop 512 1024 --batch 2 --amp --check-divisibility"""),

(CODE, """# M2 GATE — dataset validation suite (writes figures/sanity_overlays/)
!python -m tests.test_dataset --data-root $CS_ROOT 2>&1 | tail -40"""),

(CODE, """# M2 manual step — LOOK at the overlays. Mask edges must sit on object edges.
import glob
from IPython.display import Image, display
for p in sorted(glob.glob("figures/sanity_overlays/*.png"))[:4]:
    print(p); display(Image(p, width=1100))"""),

(CODE, """# M5 GATE — metric and loss unit tests (fast, no GPU needed)
!python -m tests.test_metrics && python -m tests.test_losses"""),

(CODE, """# M6 GATE — supervision cost. Decide mutation vs deep vs last_layer with numbers.
!python -m tools.supervision_cost --crop 512 1024 --batch 2 --amp --iters 10 --warmup 5"""),

(CODE, """# M4 GATE — tiny overfit. Nothing long runs until this passes.
!python -m tools.tiny_overfit --data-root $CS_ROOT --images 12 --iters 300 \\
    --batch-size 2 --crop 512 1024 --supervision deep_supervision --loss ce_dice --amp"""),

(CODE, """# M7 GATE — short sanity run (~15 min). Loss must trend down, val mIoU > chance.
!python -W ignore train_cityscapes.py --config configs/cityscapes_pvtv2b2.yaml \\
    --data-root $CS_ROOT --max-iters 300 --val-interval 300 --val-limit 50 \\
    --out-dir /kaggle/working/runs/sanity"""),

(CODE, """# M8 — primary run. max_hours stops and checkpoints before the session dies.
!python -W ignore train_cityscapes.py --config configs/cityscapes_pvtv2b2.yaml \\
    --data-root $CS_ROOT --out-dir /kaggle/working/runs/primary"""),

(CODE, """# M8 (continued) — resume in a later session. Safe to re-run any number of times.
!python -W ignore train_cityscapes.py --config configs/cityscapes_pvtv2b2.yaml \\
    --data-root $CS_ROOT --out-dir /kaggle/working/runs/primary \\
    --resume /kaggle/working/runs/primary/last.pth"""),

(CODE, """# M9 GATE — final validation. Native resolution is the standard protocol;
# fall back to --val-resize 512 1024 if it will not fit, and say so in the README.
!python -W ignore eval_cityscapes.py \\
    --checkpoint /kaggle/working/runs/primary/best_miou.pth \\
    --data-root $CS_ROOT --out-dir results --save-preds results/preds_trainids"""),

(CODE, """# M10 — qualitative panels + quantitative failure analysis
!python -W ignore visualize_cityscapes.py \\
    --checkpoint /kaggle/working/runs/primary/best_miou.pth \\
    --data-root $CS_ROOT --num-images 8 --strategy spread --out-dir figures
!python -m tools.analyze_failures"""),

(CODE, """import glob
from IPython.display import Image, display
for p in sorted(glob.glob("figures/qualitative_*.png"))[:4]:
    display(Image(p, width=1400))
display(Image("figures/iou_vs_frequency.png", width=700))
display(Image("figures/confusion_heatmap.png", width=700))"""),

(CODE, """# Optional cross-check against the official Cityscapes evaluator (needs internet)
# !pip install -q cityscapesscripts
# !python -m tools.official_eval_crosscheck --pred-dir results/preds_trainids \\
#     --data-root $CS_ROOT --custom-summary results/summary.json --run-official"""),

(CODE, """# Optional single ablation: identical config, CE only
!python -W ignore train_cityscapes.py --config configs/cityscapes_pvtv2b2_ce.yaml \\
    --data-root $CS_ROOT --out-dir /kaggle/working/runs/ce_only"""),

(CODE, """# Collect everything the submission needs into one folder
import shutil, os, json
OUT = "/kaggle/working/submission"
os.makedirs(OUT, exist_ok=True)
for d in ["results", "figures", "logs", "configs"]:
    if os.path.isdir(d):
        shutil.copytree(d, f"{OUT}/{d}", dirs_exist_ok=True)
for f in glob.glob("/kaggle/working/runs/*/train_log.csv") + \\
         glob.glob("/kaggle/working/runs/*/train_summary.json") + \\
         glob.glob("/kaggle/working/runs/*/config.json"):
    tag = f.split("/runs/")[1].replace("/", "_")
    shutil.copy(f, f"{OUT}/logs/{tag}")
print(json.load(open("results/summary.json"))["miou"])
print(sorted(os.listdir(OUT)))
# Download /kaggle/working/submission, commit it to the repo, push."""),

(MD, """## After the gates

1. Copy the numbers from `results/summary.json` and `results/per_class_iou.csv`
   into the README tables (templates in `MANUAL.md` section 24).
2. Write the failure analysis from `results/failure_analysis.json` plus the
   panels — hedged language only (`suggests`, `is consistent with`).
3. Commit `logs/`, `results/`, `figures/`, `configs/` so the run is auditable.
4. Work through `MANUAL.md` section 26 before the interview."""),
]


def build():
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        src_lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == MD:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src_lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "source": src_lines,
                          "execution_count": None, "outputs": []})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "EMCAD_Cityscapes_Kaggle.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
