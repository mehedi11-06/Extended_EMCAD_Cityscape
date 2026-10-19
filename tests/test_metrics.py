"""Unit tests for the IoU/mIoU evaluator. Run: python -m tests.test_metrics"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import ConfusionMatrix
from utils.cityscapes_labels import NUM_CLASSES, IGNORE_INDEX

C = NUM_CLASSES


def _gt(h=40, w=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, C, size=(h, w)).astype(np.int64)


def test_perfect():
    gt = _gt()
    cm = ConfusionMatrix()
    cm.update(gt, gt)
    iou = cm.iou()
    assert np.allclose(iou[~np.isnan(iou)], 1.0), iou
    assert abs(cm.miou() - 1.0) < 1e-12
    assert abs(cm.pixel_accuracy() - 1.0) < 1e-12
    print("  perfect prediction -> all IoU 1.0, mIoU 1.0  OK")


def test_all_wrong():
    gt = _gt()
    pred = (gt + 1) % C          # every pixel wrong, every class used
    cm = ConfusionMatrix()
    cm.update(pred, gt)
    iou = cm.iou()
    assert np.allclose(iou[~np.isnan(iou)], 0.0), iou
    assert cm.miou() == 0.0
    print("  all predictions wrong -> all IoU 0.0, mIoU 0.0  OK")


def test_one_class_swap():
    """Swapping exactly two classes must degrade exactly those two IoUs."""
    gt = _gt()
    a, b = 5, 7                                  # pole <-> traffic sign
    pred = gt.copy()
    pred[gt == a] = b
    pred[gt == b] = a
    cm = ConfusionMatrix()
    cm.update(pred, gt)
    iou = cm.iou()
    assert iou[a] == 0.0 and iou[b] == 0.0, (iou[a], iou[b])
    others = [i for i in range(C) if i not in (a, b) and not np.isnan(iou[i])]
    assert np.allclose(iou[others], 1.0), iou[others]
    print(f"  swap of classes {a}/{b} -> only those two degrade  OK")


def test_all_ignore():
    """An entirely ignored target must contribute nothing and not divide by 0."""
    gt = np.full((20, 20), IGNORE_INDEX, dtype=np.int64)
    pred = np.zeros((20, 20), dtype=np.int64)
    cm = ConfusionMatrix()
    cm.update(pred, gt)
    assert cm.mat.sum() == 0
    iou = cm.iou()
    assert np.all(np.isnan(iou)), iou
    assert np.isnan(cm.miou())
    print("  all-ignore target -> empty matrix, NaN mIoU, no crash  OK")


def test_mixed_ignore():
    """Ignored pixels must not change the result, whatever is predicted there."""
    gt = _gt(seed=1)
    pred = gt.copy()
    mask = np.zeros_like(gt, dtype=bool)
    mask[:10, :] = True                  # ignore the top strip
    gt_ign = gt.copy()
    gt_ign[mask] = IGNORE_INDEX

    cm_a = ConfusionMatrix(); cm_a.update(pred, gt_ign)
    pred_b = pred.copy()
    rng = np.random.default_rng(2)
    pred_b[mask] = rng.integers(0, C, size=mask.sum())   # garbage where ignored
    cm_b = ConfusionMatrix(); cm_b.update(pred_b, gt_ign)

    assert np.array_equal(cm_a.mat, cm_b.mat)
    assert abs(cm_a.miou() - 1.0) < 1e-12
    assert cm_a.mat.sum() == (~mask).sum()
    print("  mixed ignore -> identical matrix regardless of ignored predictions  OK")


def test_absent_class():
    """A class absent from the whole split is NaN and excluded from the mean."""
    gt = np.zeros((10, 10), dtype=np.int64)
    gt[:, 5:] = 1                        # only classes 0 and 1 exist
    cm = ConfusionMatrix()
    cm.update(gt, gt)
    iou = cm.iou()
    assert not np.isnan(iou[0]) and not np.isnan(iou[1])
    assert np.all(np.isnan(iou[2:]))
    assert abs(cm.miou() - 1.0) < 1e-12
    assert cm.summary()["n_valid_classes"] == 2
    print("  absent classes -> NaN, excluded from mIoU, counted in summary  OK")


def test_known_value():
    """Hand-computable case: IoU = 2/(2+1+1) = 0.5 for class 1."""
    gt = np.array([[0, 0, 1, 1]], dtype=np.int64)
    pred = np.array([[0, 1, 1, 0]], dtype=np.int64)
    cm = ConfusionMatrix()
    cm.update(pred, gt)
    iou = cm.iou()
    # class 0: TP=1 FP=1 FN=1 -> 1/3 ; class 1: TP=1 FP=1 FN=1 -> 1/3
    assert abs(iou[0] - 1 / 3) < 1e-12 and abs(iou[1] - 1 / 3) < 1e-12, iou[:2]
    assert abs(cm.miou() - 1 / 3) < 1e-12
    print("  hand-computed 4-pixel case matches 1/3  OK")


def test_accumulation():
    """Batched updates must equal a single update over the concatenation."""
    gt = _gt(h=16, w=16, seed=3)
    rng = np.random.default_rng(4)
    pred = np.where(rng.random(gt.shape) < 0.3, rng.integers(0, C, gt.shape), gt)
    one = ConfusionMatrix(); one.update(pred, gt)
    many = ConfusionMatrix()
    for i in range(0, 16, 4):
        many.update(pred[i:i + 4], gt[i:i + 4])
    assert np.array_equal(one.mat, many.mat)
    print("  incremental accumulation equals single-shot  OK")


if __name__ == "__main__":
    print("metric unit tests")
    for fn in [test_perfect, test_all_wrong, test_one_class_swap, test_all_ignore,
               test_mixed_ignore, test_absent_class, test_known_value, test_accumulation]:
        fn()
    print("ALL METRIC TESTS PASSED")
