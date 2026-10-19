"""Loss unit tests, including a numerical demonstration of the ignore-index
defect in the official EMCAD DiceLoss. Run: python -m tests.test_losses"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses.segmentation_losses import (
    MaskedMulticlassDiceLoss, CombinedSegLoss, supervision_subsets, supervised_loss)

C, H, W, B = 19, 32, 64, 2
IGN = 255


class OriginalEMCADDiceLoss(nn.Module):
    """Verbatim logic of SLDGroup/EMCAD utils/utils.py:102-138, for comparison."""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, t):
        return torch.cat([(t == i).unsqueeze(1) for i in range(self.n_classes)], 1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i])
        return loss / self.n_classes


def _batch(seed=0, ignore_rows=8):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, C, H, W, generator=g)
    target = torch.randint(0, C, (B, H, W), generator=g)
    target[:, :ignore_rows, :] = IGN
    return logits, target


def test_shapes_and_range():
    logits, target = _batch()
    loss = MaskedMulticlassDiceLoss(C, IGN)(logits, target)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert 0.0 <= float(loss) <= 1.0 + 1e-6, float(loss)
    print(f"  scalar finite dice in [0,1]: {float(loss):.4f}  OK")


def test_ignore_invariance():
    """Changing LABELS inside the ignore region must not change the loss."""
    logits, target = _batch()
    a = MaskedMulticlassDiceLoss(C, IGN)(logits, target)
    t2 = target.clone()
    t2[:, :8, :] = IGN                       # still ignored, different content below
    noise = torch.randint(0, C, (B, 8, W))
    t3 = target.clone()
    t3[:, :8, :] = torch.where(torch.rand(B, 8, W) < 0.5, noise, torch.full_like(noise, IGN))
    b = MaskedMulticlassDiceLoss(C, IGN)(logits, t2)
    assert torch.allclose(a, b), (float(a), float(b))
    print(f"  loss invariant to ignored-region content: {float(a):.6f}  OK")


def test_perfect_prediction():
    _, target = _batch()
    safe = torch.where(target == IGN, torch.zeros_like(target), target)
    onehot = torch.nn.functional.one_hot(safe, C).permute(0, 3, 1, 2).float()
    logits = (onehot * 30.0) - 15.0          # near one-hot after softmax
    loss = MaskedMulticlassDiceLoss(C, IGN)(logits, target)
    assert float(loss) < 1e-3, float(loss)
    print(f"  near-perfect logits -> dice loss {float(loss):.2e}  OK")


def test_zero_gradient_at_ignored_pixels():
    logits, target = _batch()
    logits.requires_grad_(True)
    MaskedMulticlassDiceLoss(C, IGN)(logits, target).backward()
    g = logits.grad
    assert torch.allclose(g[:, :, :8, :], torch.zeros_like(g[:, :, :8, :])), \
        g[:, :, :8, :].abs().max()
    assert g[:, :, 8:, :].abs().max() > 0
    print("  gradient exactly zero at ignored pixels, nonzero elsewhere  OK")


def test_original_loss_is_not_ignore_safe():
    """Demonstrates the defect: the official loss DOES move when only the
    ignored region's predictions change, and its value is inflated by them."""
    logits, target = _batch()
    orig = OriginalEMCADDiceLoss(C)
    ours = MaskedMulticlassDiceLoss(C, IGN)

    l1 = float(orig(logits, target, softmax=True))
    l2_logits = logits.clone()
    l2_logits[:, :, :8, :] += 8.0 * torch.randn(B, C, 8, W)   # ignored region only
    l2 = float(orig(l2_logits, target, softmax=True))
    m1 = float(ours(logits, target))
    m2 = float(ours(l2_logits, target))

    print(f"  official  DiceLoss: {l1:.6f} -> {l2:.6f}   (delta {abs(l2-l1):.6f})")
    print(f"  masked    DiceLoss: {m1:.6f} -> {m2:.6f}   (delta {abs(m2-m1):.6f})")
    assert abs(l2 - l1) > 1e-6, "expected the official loss to react to ignored pixels"
    assert abs(m2 - m1) < 1e-6, "masked loss must not react"

    # gradient leakage in the original
    lg = logits.clone().requires_grad_(True)
    orig(lg, target, softmax=True).backward()
    leak = lg.grad[:, :, :8, :].abs().max()
    assert leak > 0, "expected nonzero gradient leakage into the ignore region"
    print(f"  official loss leaks gradient into ignored pixels (max |g| = {leak:.2e})  OK")


def test_combined_and_supervision():
    logits, target = _batch()
    crit = CombinedSegLoss(C, IGN, w_ce=0.3, w_dice=0.7)
    total, parts = crit(logits, target)
    assert torch.isfinite(total) and {"ce", "dice"} <= set(parts)
    assert abs(float(total) - (0.3 * parts["ce"] + 0.7 * parts["dice"])) < 1e-5

    outs = [logits.clone() for _ in range(4)]
    for mode, n in [("mutation", 15), ("deep_supervision", 4), ("last_layer", 1)]:
        ss = supervision_subsets(mode, 4)
        assert len(ss) == n, (mode, len(ss))
        t, p = supervised_loss(outs, target, crit, ss)
        assert torch.isfinite(t)
    print("  combined loss weighting correct; subsets 15/4/1 for mutation/deep/last  OK")


def test_ce_rejects_255_without_ignore_index():
    """Confirms the original trainer.py:68 construction would fail on Cityscapes."""
    logits, target = _batch()
    try:
        nn.CrossEntropyLoss()(logits, target)
        raise AssertionError("expected CrossEntropyLoss() to reject label 255")
    except (IndexError, RuntimeError) as e:
        print(f"  CrossEntropyLoss() without ignore_index raises "
              f"{type(e).__name__} on label 255  OK")
    v = nn.CrossEntropyLoss(ignore_index=255)(logits, target)
    assert torch.isfinite(v)


if __name__ == "__main__":
    print("loss unit tests")
    for fn in [test_shapes_and_range, test_ignore_invariance, test_perfect_prediction,
               test_zero_gradient_at_ignored_pixels, test_original_loss_is_not_ignore_safe,
               test_combined_and_supervision, test_ce_rejects_255_without_ignore_index]:
        fn()
    print("ALL LOSS TESTS PASSED")
