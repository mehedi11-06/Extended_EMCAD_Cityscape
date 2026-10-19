"""Cityscapes dataset for 19-class semantic segmentation.

Contract (one sample):
    image : torch.float32 [3, H, W]   ImageNet-normalized
    label : torch.int64   [H, W]      values in {0..18} u {255}
    name  : str                       e.g. "frankfurt_000000_000294"

Design notes
------------
* Masks are read from *_gtFine_labelTrainIds.png when present. When a mirror
  ships only *_gtFine_labelIds.png, the official labelId->trainId table in
  utils/cityscapes_labels.py is applied through a 256-entry LUT. Which source
  was used is recorded on the dataset object (`self.mask_kind`) so it can be
  reported in the README.
* Every geometric op is applied to image and mask from the SAME random draw
  inside one function. Image uses bilinear, mask uses nearest, always.
* Crop padding uses the ignore index for the mask (never class 0) and the
  ImageNet mean for the image.
"""

import os
import glob
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.cityscapes_labels import (
    NUM_CLASSES, IGNORE_INDEX, build_id_to_trainid_lut,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_cityscapes_root(search_roots=("/kaggle/input", "./data")):
    """Locate a directory that contains both leftImg8bit/ and gtFine/.

    Kaggle dataset mounts vary, so the layout is discovered rather than assumed.
    Returns the first matching root, or None.
    """
    candidates = []
    for base in search_roots:
        if not os.path.isdir(base):
            continue
        for depth in ("", "*/", "*/*/", "*/*/*/"):
            for p in glob.glob(os.path.join(base, depth)):
                if (os.path.isdir(os.path.join(p, "leftImg8bit"))
                        and os.path.isdir(os.path.join(p, "gtFine"))):
                    candidates.append(os.path.normpath(p))
    return candidates[0] if candidates else None


def mask_path_for(image_path):
    """leftImg8bit path -> (labelTrainIds path, labelIds path).

    The pairing is a pure string substitution in both the directory component
    and the filename suffix, so it can be asserted rather than trusted.
    """
    p = image_path.replace("/leftImg8bit/", "/gtFine/", 1)
    base = p[: -len("_leftImg8bit.png")]
    return base + "_gtFine_labelTrainIds.png", base + "_gtFine_labelIds.png"


class CityscapesSegmentation(Dataset):
    def __init__(self, root, split="train", crop_size=(512, 1024),
                 scale_range=(0.5, 2.0), hflip=True, color_jitter=True,
                 augment=None, val_resize=None, limit=None, seed=None):
        """
        root        : directory containing leftImg8bit/ and gtFine/
        split       : "train" or "val"
        crop_size   : (H, W) training crop; both must be divisible by 32
        scale_range : random scale factors applied before cropping (train only)
        augment     : None -> True for train, False for val. Set False to
                      disable all augmentation (used by the tiny-overfit test).
        val_resize  : (H, W) to resize val images to, or None for full 1024x2048
        limit       : keep only the first N samples (debugging / tiny-overfit)
        """
        self.root = root
        self.split = split
        self.crop_size = tuple(crop_size)
        self.scale_range = scale_range
        self.hflip = hflip
        self.color_jitter = color_jitter
        self.augment = (split == "train") if augment is None else augment
        self.val_resize = tuple(val_resize) if val_resize else None
        self.lut = build_id_to_trainid_lut()

        pattern = os.path.join(root, "leftImg8bit", split, "*", "*_leftImg8bit.png")
        self.images = sorted(glob.glob(pattern))
        if not self.images:
            raise FileNotFoundError(f"no images matched {pattern}")

        self.masks, missing, kinds = [], [], set()
        for ip in self.images:
            train_ids, label_ids = mask_path_for(ip)
            if os.path.exists(train_ids):
                self.masks.append(train_ids)
                kinds.add("labelTrainIds")
            elif os.path.exists(label_ids):
                self.masks.append(label_ids)
                kinds.add("labelIds")
            else:
                missing.append(ip)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images have no mask, e.g. {missing[0]} -> "
                f"{mask_path_for(missing[0])[0]}")
        self.mask_kind = "+".join(sorted(kinds))

        if limit is not None:
            rng = random.Random(seed if seed is not None else 0)
            idx = list(range(len(self.images)))
            rng.shuffle(idx)
            idx = sorted(idx[:limit])
            self.images = [self.images[i] for i in idx]
            self.masks = [self.masks[i] for i in idx]

        for h in self.crop_size:
            assert h % 32 == 0, (
                "EMCADNet upsamples with fixed scale factors 32/16/8/4 "
                "(lib/networks.py:107-110), so crop H and W must be divisible "
                f"by 32; got {self.crop_size}")

    def __len__(self):
        return len(self.images)

    def name(self, i):
        return os.path.basename(self.images[i])[: -len("_leftImg8bit.png")]

    def _load(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        m = Image.open(self.masks[i])
        mask = np.array(m)
        if self.masks[i].endswith("_labelIds.png"):
            mask = self.lut[mask]                      # official id -> trainId
        return img, Image.fromarray(mask.astype(np.uint8))

    def _joint_transform(self, img, mask):
        """All geometric ops, image bilinear / mask nearest, one random draw."""
        ch, cw = self.crop_size

        if self.augment:
            # --- random scale (applied to the full frame) ---
            s = random.uniform(*self.scale_range)
            w, h = img.size
            nw, nh = int(round(w * s)), int(round(h * s))
            img = img.resize((nw, nh), Image.BILINEAR)
            mask = mask.resize((nw, nh), Image.NEAREST)

            # --- pad if smaller than the crop: mask with IGNORE, never 0 ---
            pw, ph = max(cw - nw, 0), max(ch - nh, 0)
            if pw or ph:
                fill = tuple(int(255 * v) for v in IMAGENET_MEAN)
                img = _pad(img, pw, ph, fill)
                mask = _pad(mask, pw, ph, IGNORE_INDEX)
                nw, nh = img.size

            # --- random crop (same box for both) ---
            x0 = random.randint(0, nw - cw)
            y0 = random.randint(0, nh - ch)
            box = (x0, y0, x0 + cw, y0 + ch)
            img, mask = img.crop(box), mask.crop(box)

            # --- horizontal flip (same decision for both) ---
            if self.hflip and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # --- photometric jitter: IMAGE ONLY ---
            if self.color_jitter:
                img = _jitter(img)
        else:
            if self.split == "train":
                # deterministic centre crop for the tiny-overfit experiment
                w, h = img.size
                x0, y0 = (w - cw) // 2, (h - ch) // 2
                box = (x0, y0, x0 + cw, y0 + ch)
                img, mask = img.crop(box), mask.crop(box)
            elif self.val_resize is not None:
                vh, vw = self.val_resize
                img = img.resize((vw, vh), Image.BILINEAR)
                mask = mask.resize((vw, vh), Image.NEAREST)
        return img, mask

    def __getitem__(self, i):
        img, mask = self._load(i)
        img, mask = self._joint_transform(img, mask)

        x = torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std

        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        return {"image": x, "label": y, "name": self.name(i)}


def _pad(pil, pw, ph, fill):
    from PIL import ImageOps
    # pad right/bottom only; keeps the crop box arithmetic simple
    return ImageOps.expand(pil, border=(0, 0, pw, ph), fill=fill)


def _jitter(img, brightness=0.3, contrast=0.3, saturation=0.3):
    from PIL import ImageEnhance
    for enh, amt in ((ImageEnhance.Brightness, brightness),
                     (ImageEnhance.Contrast, contrast),
                     (ImageEnhance.Color, saturation)):
        if amt > 0:
            img = enh(img).enhance(random.uniform(1 - amt, 1 + amt))
    return img


def denormalize(x):
    """[3,H,W] normalized tensor -> [H,W,3] uint8, for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (x.detach().cpu() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
