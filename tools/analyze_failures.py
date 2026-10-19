"""Quantitative support for the failure analysis.

Reads results/confusion_matrix.npy and results/class_frequency.csv and writes:

  results/failure_analysis.json       group mIoU, top confusion pairs per class,
                                      IoU-vs-frequency correlation
  results/thin_vs_large.csv           the grouped comparison the assignment asks for
  figures/iou_vs_frequency.png        scatter, log-x
  figures/confusion_heatmap.png       row-normalized 19x19

Every number here is derived from measurements. Nothing is asserted about
causes; the JSON records evidence, the README states hypotheses in hedged
language.

    python -m tools.analyze_failures
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cityscapes_labels import CLASS_NAMES, CLASS_GROUPS, NUM_CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm", default="results/confusion_matrix.npy")
    ap.add_argument("--freq", default="results/class_frequency.csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    m = np.load(args.cm).astype(np.float64)
    tp = np.diag(m)
    fn = m.sum(1) - tp
    fp = m.sum(0) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        recall = np.where(m.sum(1) > 0, tp / m.sum(1), np.nan)
        precision = np.where(m.sum(0) > 0, tp / m.sum(0), np.nan)

    freq = {}
    if os.path.exists(args.freq):
        with open(args.freq) as f:
            for r in csv.DictReader(f):
                freq[r["class_name"]] = float(r["pixel_fraction"])

    # top confusions: where does each class's ground truth actually go?
    confusions = {}
    for c in range(NUM_CLASSES):
        row = m[c].copy()
        row[c] = 0
        order = np.argsort(-row)[: args.top_k]
        tot = m[c].sum()
        confusions[CLASS_NAMES[c]] = [
            {"predicted_as": CLASS_NAMES[j],
             "share_of_class_gt_pixels": round(float(row[j] / tot), 5) if tot else None}
            for j in order if row[j] > 0]

    groups = {}
    for g, ids in CLASS_GROUPS.items():
        v = iou[ids]
        groups[g] = {
            "classes": [CLASS_NAMES[i] for i in ids],
            "mean_iou": float(np.nanmean(v)) if np.any(~np.isnan(v)) else None,
            "mean_pixel_fraction": (float(np.mean([freq.get(CLASS_NAMES[i], np.nan) for i in ids]))
                                    if freq else None),
        }

    corr = None
    if freq:
        x = np.array([freq.get(n, np.nan) for n in CLASS_NAMES])
        ok = ~np.isnan(x) & ~np.isnan(iou)
        if ok.sum() > 2:
            corr = float(np.corrcoef(np.log10(x[ok]), iou[ok])[0, 1])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "thin_vs_large.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "trainId", "class_name", "iou", "precision", "recall",
                    "pixel_fraction"])
        for g, ids in CLASS_GROUPS.items():
            for i in ids:
                w.writerow([g, i, CLASS_NAMES[i],
                            "" if np.isnan(iou[i]) else f"{iou[i]:.4f}",
                            "" if np.isnan(precision[i]) else f"{precision[i]:.4f}",
                            "" if np.isnan(recall[i]) else f"{recall[i]:.4f}",
                            f"{freq.get(CLASS_NAMES[i], float('nan')):.6f}" if freq else ""])

    out = {"per_class": {CLASS_NAMES[i]: {
                "iou": None if np.isnan(iou[i]) else float(iou[i]),
                "precision": None if np.isnan(precision[i]) else float(precision[i]),
                "recall": None if np.isnan(recall[i]) else float(recall[i]),
                "pixel_fraction": freq.get(CLASS_NAMES[i]),
            } for i in range(NUM_CLASSES)},
           "group_summary": groups,
           "top_confusions": confusions,
           "log_frequency_vs_iou_pearson_r": corr,
           "note": ("Correlation is descriptive. It is consistent with a "
                    "frequency/size effect but does not establish causation.")}
    with open(os.path.join(args.out_dir, "failure_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.fig_dir, exist_ok=True)

        if freq:
            fig, ax = plt.subplots(figsize=(7.5, 5))
            colors = {"large_region": "#2b6cb0", "medium_object": "#8a6d1f",
                      "small_thin": "#b03030"}
            for g, ids in CLASS_GROUPS.items():
                ax.scatter([freq[CLASS_NAMES[i]] for i in ids],
                           [iou[i] for i in ids], label=g, s=48, color=colors[g])
            for i, n in enumerate(CLASS_NAMES):
                if n in freq and not np.isnan(iou[i]):
                    ax.annotate(n, (freq[n], iou[i]), fontsize=7,
                                xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_xlabel("train-set pixel fraction (log)")
            ax.set_ylabel("validation IoU")
            ax.set_title(f"Per-class IoU vs class frequency"
                         + (f"  (r = {corr:.2f})" if corr is not None else ""))
            ax.grid(alpha=.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.fig_dir, "iou_vs_frequency.png"), dpi=150)
            plt.close(fig)

        rown = m / np.clip(m.sum(1, keepdims=True), 1, None)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(rown, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(19)); ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=8)
        ax.set_yticks(range(19)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
        ax.set_title("Row-normalized confusion matrix")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(args.fig_dir, "confusion_heatmap.png"), dpi=150)
        plt.close(fig)
        print(f"figures written to {args.fig_dir}")
    except Exception as e:
        print(f"[warn] figures skipped: {type(e).__name__}: {e}")

    print(json.dumps({"group_summary": groups,
                      "log_frequency_vs_iou_pearson_r": corr}, indent=2))
    print(f"\nwrote {args.out_dir}/failure_analysis.json and thin_vs_large.csv")


if __name__ == "__main__":
    main()
