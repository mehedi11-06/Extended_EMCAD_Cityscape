"""Loss functions for 19-class Cityscapes segmentation with ignore_index=255.

Why this file exists instead of reusing EMCAD's utils/utils.py:DiceLoss:

    _one_hot_encoder() builds (target == i) for i in 0..C-1, so a pixel
    labelled 255 gets an all-zero one-hot row. That is fine for the target
    side, but in _dice_loss the term

        z_sum = torch.sum(score * score)

    still sums the predicted probabilities at those pixels. The ignored pixel
    adds nothing to the numerator and something positive to the denominator,
    so the optimizer is rewarded for pushing every class probability toward 0
    at ignored pixels -- impossible under softmax, and a systematic bias
    because Cityscapes ignore regions (ego vehicle, border) sit in the same
    place in every frame.

MaskedMulticlassDiceLoss multiplies both probs and one-hot by the valid mask
before any reduction, so ignored pixels contribute exactly zero everywhere and
receive zero gradient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMulticlassDiceLoss(nn.Module):
    """Soft multiclass Dice that honours an ignore index.

    Defaults mirror EMCAD (squared denominator, batch-level reduction,
    smooth=1e-5) so results stay comparable to the baseline formulation.
    """

    def __init__(self, num_classes=19, ignore_index=255, smooth=1e-5,
                 squared_denominator=True, skip_absent_classes=True):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.squared_denominator = squared_denominator
        self.skip_absent_classes = skip_absent_classes

    def forward(self, logits, target):
        assert logits.dim() == 4 and target.dim() == 3, (logits.shape, target.shape)
        assert logits.size(0) == target.size(0)
        assert logits.shape[2:] == target.shape[1:], (logits.shape, target.shape)
        assert logits.size(1) == self.num_classes

        valid = target != self.ignore_index                       # [B,H,W]
        # F.one_hot would raise on 255; remap then erase with the mask.
        safe = torch.where(valid, target, torch.zeros_like(target))
        onehot = F.one_hot(safe.long(), self.num_classes)          # [B,H,W,C]
        onehot = onehot.permute(0, 3, 1, 2).to(logits.dtype)       # [B,C,H,W]

        probs = torch.softmax(logits.float(), dim=1).to(logits.dtype)
        m = valid.unsqueeze(1).to(logits.dtype)
        probs = probs * m
        onehot = onehot * m

        dims = (0, 2, 3)                                           # batch-level
        intersect = (probs * onehot).sum(dims)                     # [C]
        if self.squared_denominator:
            denom = (probs * probs).sum(dims) + (onehot * onehot).sum(dims)
        else:
            denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * intersect + self.smooth) / (denom + self.smooth)

        if self.skip_absent_classes:
            present = onehot.sum(dims) > 0
            if bool(present.any()):
                return 1.0 - dice[present].mean()
        return 1.0 - dice.mean()


class MaskedFocalLoss(nn.Module):
    """Focal loss with ignore index. Provided for completeness; not used in
    the primary experiment (see MANUAL.md section 9)."""

    def __init__(self, gamma=2.0, ignore_index=255, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.alpha = alpha

    def forward(self, logits, target):
        ce = F.cross_entropy(logits.float(), target.long(),
                             ignore_index=self.ignore_index,
                             reduction="none", weight=self.alpha)   # [B,H,W]
        valid = (target != self.ignore_index)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        n = valid.sum()
        return loss.sum() / n.clamp(min=1) if n > 0 else loss.sum() * 0.0


class CombinedSegLoss(nn.Module):
    """w_ce * CE + w_dice * Dice (+ w_focal * Focal).

    EMCAD's trainer.py:103 uses w_ce=0.3, w_dice=0.7; those are the defaults
    here so the formulation stays faithful to the baseline.
    """

    def __init__(self, num_classes=19, ignore_index=255,
                 w_ce=0.3, w_dice=0.7, w_focal=0.0, gamma=2.0,
                 squared_denominator=True, skip_absent_classes=True):
        super().__init__()
        self.w_ce, self.w_dice, self.w_focal = w_ce, w_dice, w_focal
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = MaskedMulticlassDiceLoss(
            num_classes, ignore_index, squared_denominator=squared_denominator,
            skip_absent_classes=skip_absent_classes) if w_dice > 0 else None
        self.focal = MaskedFocalLoss(gamma, ignore_index) if w_focal > 0 else None

    def forward(self, logits, target):
        """Returns (total, parts_dict) so each component can be logged."""
        parts = {}
        total = logits.new_zeros(())
        if self.w_ce > 0:
            l = self.ce(logits.float(), target.long())
            parts["ce"] = float(l.detach())
            total = total + self.w_ce * l
        if self.dice is not None:
            l = self.dice(logits, target)
            parts["dice"] = float(l.detach())
            total = total + self.w_dice * l
        if self.focal is not None:
            l = self.focal(logits, target)
            parts["focal"] = float(l.detach())
            total = total + self.w_focal * l
        return total, parts


def supervision_subsets(mode, n_outputs):
    """Reproduce EMCAD's trainer.py:91-100 subset construction.

    mutation          -> all non-empty subsets of the output indices (15 for 4)
    deep_supervision  -> one subset per output ([[0],[1],[2],[3]])
    last_layer        -> [[-1]], i.e. the final (finest) head only
    """
    idxs = list(range(n_outputs))
    if mode == "mutation":
        subsets = [[]]
        for i in idxs:                       # standard powerset construction
            subsets += [s + [i] for s in subsets]
        return [s for s in subsets if s]     # drop the empty set (trainer.py:107)
    if mode == "deep_supervision":
        return [[i] for i in idxs]
    if mode == "last_layer":
        return [[-1]]
    raise ValueError(f"unknown supervision mode: {mode}")


def supervised_loss(outputs, target, criterion, subsets):
    """Apply `criterion` over EMCAD's subset scheme.

    Within a subset the LOGITS are summed before the loss, exactly as
    trainer.py:109-110 does (`iout += P[s[idx]]`).
    """
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    total = None
    agg = {}
    for s in subsets:
        iout = outputs[s[0]]
        for k in s[1:]:
            iout = iout + outputs[k]
        l, parts = criterion(iout, target)
        total = l if total is None else total + l
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
    n = max(len(subsets), 1)
    return total, {k: v / n for k, v in agg.items()}
