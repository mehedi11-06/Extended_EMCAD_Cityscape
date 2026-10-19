"""Confusion-matrix IoU / mIoU for Cityscapes, ignoring label 255.

Convention: rows = ground truth, columns = prediction.

    TP_c  = M[c, c]
    FN_c  = M[c, :].sum() - M[c, c]      GT is c, predicted something else
    FP_c  = M[:, c].sum() - M[c, c]      predicted c, GT was something else
    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    mIoU  = mean over classes PRESENT in the ground truth of the full split

Absent-class policy
-------------------
A class absent from one batch is simply not special: the matrix accumulates
over the whole validation set, so presence is only decided at the end.
A class absent from the ENTIRE validation set has TP+FP+FN == 0; its IoU is
reported as NaN and excluded from the mean, which is what the official
Cityscapes evaluation does. All 19 Cityscapes classes do appear in the
official val split, so in a correct run nothing should be excluded -- if
something is, that is a signal worth investigating, and `n_valid_classes`
records it.
"""

import numpy as np
import torch

from utils.cityscapes_labels import CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX, CLASS_GROUPS


class ConfusionMatrix:
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        self.n = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((self.n, self.n), dtype=np.int64)

    def reset(self):
        self.mat[:] = 0

    @torch.no_grad()
    def update(self, pred, target):
        """pred, target: [B,H,W] (or [H,W]) integer tensors or arrays."""
        pred = _to_numpy(pred).reshape(-1)
        target = _to_numpy(target).reshape(-1)
        assert pred.shape == target.shape, (pred.shape, target.shape)

        valid = (target != self.ignore_index)
        # Guard against out-of-range predictions rather than silently wrapping.
        p, t = pred[valid].astype(np.int64), target[valid].astype(np.int64)
        if p.size == 0:
            return
        assert p.min() >= 0 and p.max() < self.n, f"pred out of range: {p.min()}..{p.max()}"
        assert t.min() >= 0 and t.max() < self.n, f"target out of range: {t.min()}..{t.max()}"

        self.mat += np.bincount(self.n * t + p,
                                minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self):
        """Per-class IoU as float64 array of length n; NaN for absent classes."""
        m = self.mat.astype(np.float64)
        tp = np.diag(m)
        fn = m.sum(axis=1) - tp
        fp = m.sum(axis=0) - tp
        denom = tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(denom > 0, tp / denom, np.nan)
        return iou

    def miou(self):
        iou = self.iou()
        return float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan")

    def pixel_accuracy(self):
        m = self.mat.astype(np.float64)
        return float(np.diag(m).sum() / m.sum()) if m.sum() > 0 else float("nan")

    def summary(self, class_names=None):
        names = class_names or CLASS_NAMES
        iou = self.iou()
        groups = {}
        for g, ids in CLASS_GROUPS.items():
            vals = iou[ids]
            groups[g] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")
        return {
            "per_class_iou": {names[i]: (None if np.isnan(v) else float(v))
                              for i, v in enumerate(iou)},
            "miou": self.miou(),
            "pixel_accuracy": self.pixel_accuracy(),
            "n_valid_classes": int(np.sum(~np.isnan(iou))),
            "group_miou": groups,
        }

    def to_csv(self, path, class_names=None):
        import csv
        names = class_names or CLASS_NAMES
        iou = self.iou()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trainId", "class_name", "iou", "tp", "fp", "fn"])
            m = self.mat.astype(np.int64)
            tp = np.diag(m)
            fn = m.sum(axis=1) - tp
            fp = m.sum(axis=0) - tp
            for i, n in enumerate(names):
                v = iou[i]
                w.writerow([i, n, "" if np.isnan(v) else f"{v:.6f}",
                            int(tp[i]), int(fp[i]), int(fn[i])])
        return path

    def save_matrix(self, path):
        np.save(path, self.mat)
        return path


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
