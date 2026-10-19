"""Milestone 2 gate: the Cityscapes dataset validation suite.

Eight tests, each with a pass condition and a failure interpretation printed
alongside the result, so the log file is itself the evidence that labels were
handled correctly.

    python -m tests.test_dataset --data-root /kaggle/input/<ds>/cityscapes
    python -m tests.test_dataset --synthetic      # exercise the code with no data
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.cityscapes import (
    CityscapesSegmentation, find_cityscapes_root, mask_path_for, denormalize)
from utils.cityscapes_labels import (
    CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX, colorize,
    verify_against_cityscapesscripts)

EXPECTED = {"train": 2975, "val": 500, "test": 1525}
RESULTS = []


def check(name, ok, detail="", on_fail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok and on_fail:
        print(f"       interpretation: {on_fail}")
    return ok


# --------------------------------------------------------------------------- #
def t0_label_table():
    ok, msg = verify_against_cityscapesscripts()
    if ok is None:
        print(f"[SKIP] label table cross-check -- {msg}")
    else:
        check("label table matches cityscapesscripts", ok, msg,
              "The built-in labelId->trainId table disagrees with the official "
              "one. Do not train until resolved.")


def t1_file_count(root, strict):
    for split, exp in (("train", EXPECTED["train"]), ("val", EXPECTED["val"])):
        n = len(glob.glob(os.path.join(root, "leftImg8bit", split, "*", "*_leftImg8bit.png")))
        ok = (n == exp) if strict else (n > 0)
        check(f"file count {split}", ok, f"{n} images (official {exp})",
              "A short count means a truncated or partial dataset mirror: the "
              "experiment would silently train on less data. A long count usually "
              "means test/ was globbed too, or non-leftImg8bit files matched.")


def t2_pairing(root):
    missing = []
    for split in ("train", "val"):
        for ip in glob.glob(os.path.join(root, "leftImg8bit", split, "*", "*_leftImg8bit.png")):
            tid, lid = mask_path_for(ip)
            if not (os.path.exists(tid) or os.path.exists(lid)):
                missing.append(ip)
    check("every image has a mask", not missing,
          f"{len(missing)} unpaired" + (f", e.g. {missing[0]}" if missing else ""),
          "Either labelTrainIds were never generated (run tools/make_train_ids.py) "
          "or the path substitution rule does not match this mirror's layout.")


def t3_unique_values(root, n=100):
    masks = (glob.glob(os.path.join(root, "gtFine", "train", "*", "*_labelTrainIds.png"))
             or glob.glob(os.path.join(root, "gtFine", "train", "*", "*_labelIds.png")))
    kind = "labelTrainIds" if masks and masks[0].endswith("_labelTrainIds.png") else "labelIds"
    vals = set()
    for p in random.sample(masks, min(n, len(masks))):
        vals |= set(np.unique(np.array(Image.open(p))).tolist())
    allowed = set(range(NUM_CLASSES)) | {IGNORE_INDEX} if kind == "labelTrainIds" else set(range(34))
    check(f"raw mask values ({kind})", vals <= allowed,
          f"{sorted(vals)[:12]}{'...' if len(vals) > 12 else ''}",
          "Values 19-33 in a file you believe is labelTrainIds means you are "
          "reading labelIds. Values above 255 or a float dtype means the colour "
          "PNG was opened instead of the label PNG.")


def t4_ignore_fraction(root, n=30):
    ds = CityscapesSegmentation(root, "train", augment=False, limit=n, seed=0)
    fracs = []
    for i in range(len(ds)):
        _, m = ds._load(i)
        fracs.append(float((np.array(m) == IGNORE_INDEX).mean()))
    mean = float(np.mean(fracs))
    check("ignore-pixel fraction plausible", 0.005 < mean < 0.5,
          f"mean {100*mean:.2f}% (min {100*min(fracs):.2f}%, max {100*max(fracs):.2f}%)",
          "Near 0% suggests the mask is not a trainIds map. Above 50% suggests the "
          "wrong file, or image/mask misalignment.")


def t5_dimensions(root, n=50):
    bad = []
    imgs = glob.glob(os.path.join(root, "leftImg8bit", "train", "*", "*_leftImg8bit.png"))
    for ip in random.sample(imgs, min(n, len(imgs))):
        tid, lid = mask_path_for(ip)
        mp = tid if os.path.exists(tid) else lid
        if Image.open(ip).size != Image.open(mp).size:
            bad.append(ip)
    check("image/mask dimensions agree", not bad, f"{len(bad)} mismatched of {min(n,len(imgs))}",
          "A size mismatch on a specific frame almost always means the pairing "
          "matched the wrong city or frame id.")


def t6_augmentation_alignment(root, out_dir="figures/sanity_overlays", n=8):
    """Writes RGB | colorized mask | overlay triptychs for manual inspection."""
    os.makedirs(out_dir, exist_ok=True)
    ds = CityscapesSegmentation(root, "train", crop_size=(512, 1024), limit=n, seed=0)
    shapes_ok, wrote = True, []
    for i in range(len(ds)):
        s = ds[i]
        if tuple(s["image"].shape) != (3, 512, 1024) or tuple(s["label"].shape) != (512, 1024):
            shapes_ok = False
        rgb = denormalize(s["image"])
        col = colorize(s["label"].numpy())
        ov = (0.6 * rgb + 0.4 * col).astype(np.uint8)
        p = os.path.join(out_dir, f"{s['name']}_overlay.png")
        Image.fromarray(np.concatenate([rgb, col, ov], axis=1)).save(p)
        wrote.append(p)
    check("augmented sample shapes", shapes_ok, f"wrote {len(wrote)} overlays to {out_dir}",
          "Shape mismatch means the crop/pad logic is wrong.")

    # crops must differ between draws (augmentation is actually live)
    a, b = ds[0]["image"], ds[0]["image"]
    check("augmentation is stochastic", not np.allclose(a.numpy(), b.numpy()),
          "two draws of index 0 differ",
          "Identical draws mean augmentation is disabled or the RNG is frozen.")
    print("       MANUAL STEP: open the overlays. Mask boundaries must sit on object "
          "boundaries. A constant offset = crop desync; a mirrored mask = flip applied "
          "to one side only; a scaled mask = separate resize calls.")


def t7_post_transform_integrity(root, n=200):
    ds = CityscapesSegmentation(root, "train", crop_size=(512, 1024),
                                limit=min(n, 200), seed=1)
    bad = []
    import torch
    for i in range(len(ds)):
        lab = ds[i]["label"]
        if lab.dtype != torch.int64:
            bad.append(("dtype", lab.dtype))
            break
        u = torch.unique(lab)
        if not bool(((u <= NUM_CLASSES - 1) | (u == IGNORE_INDEX)).all()):
            bad.append((ds.name(i), u.tolist()))
    check("post-transform label integrity", not bad, f"checked {len(ds)} augmented samples",
          "Any value in 19..254 is proof that bilinear interpolation touched the "
          "mask somewhere in the transform chain. Nearest preserves the value set "
          "exactly, which is why this test detects it.")


def t8_class_frequency(root, n=300, out="results/class_frequency.csv"):
    import csv
    masks = (glob.glob(os.path.join(root, "gtFine", "train", "*", "*_labelTrainIds.png"))
             or glob.glob(os.path.join(root, "gtFine", "train", "*", "*_labelIds.png")))
    ds = CityscapesSegmentation(root, "train", augment=False, limit=1, seed=0)
    counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)
    for p in random.sample(masks, min(n, len(masks))):
        m = np.array(Image.open(p))
        if p.endswith("_labelIds.png"):
            m = ds.lut[m]
        m = np.where(m == IGNORE_INDEX, NUM_CLASSES, m)
        counts += np.bincount(m.ravel(), minlength=NUM_CLASSES + 1)[: NUM_CLASSES + 1]
    total = counts[:NUM_CLASSES].sum()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trainId", "class_name", "pixel_count", "pixel_fraction"])
        for i, nme in enumerate(CLASS_NAMES):
            w.writerow([i, nme, int(counts[i]), f"{counts[i]/total:.8f}"])
    zero = [CLASS_NAMES[i] for i in range(NUM_CLASSES) if counts[i] == 0]
    check("all 19 classes present in sample", not zero,
          f"{out}; rarest = {CLASS_NAMES[int(np.argmin(counts[:NUM_CLASSES]))]} "
          f"({100*counts[:NUM_CLASSES].min()/total:.4f}%), "
          f"ignored {100*counts[NUM_CLASSES]/counts.sum():.2f}%",
          f"Zero-count classes {zero} mean either too small a sample or a remap bug.")
    order = np.argsort(-counts[:NUM_CLASSES])
    print("       frequency order: " + " > ".join(CLASS_NAMES[i] for i in order[:6])
          + " ... " + " > ".join(CLASS_NAMES[i] for i in order[-4:]))


def make_synthetic(root, n=4):
    """Tiny fake Cityscapes tree so the pipeline can be exercised without data."""
    rng = np.random.default_rng(0)
    for split, k in (("train", n), ("val", 2)):
        for i in range(k):
            city = "synthcity"
            idir = os.path.join(root, "leftImg8bit", split, city)
            mdir = os.path.join(root, "gtFine", split, city)
            os.makedirs(idir, exist_ok=True)
            os.makedirs(mdir, exist_ok=True)
            H, W = 256, 512
            img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            m = np.zeros((H, W), dtype=np.uint8)
            for c in range(NUM_CLASSES):                 # horizontal bands
                m[c * H // NUM_CLASSES:(c + 1) * H // NUM_CLASSES] = c
            # ignore box in the lower centre, like the ego-vehicle region,
            # placed so it does not erase any band entirely
            m[-H // 10:, W // 4: 3 * W // 4] = IGNORE_INDEX
            base = f"{city}_{split}_{i:06d}"
            Image.fromarray(img).save(os.path.join(idir, base + "_leftImg8bit.png"))
            Image.fromarray(m).save(os.path.join(mdir, base + "_gtFine_labelTrainIds.png"))
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-strict-count", action="store_true",
                    help="allow non-official file counts (synthetic/subset data)")
    args = ap.parse_args()

    if args.synthetic:
        root = make_synthetic("/tmp/synthetic_cityscapes")
        strict = False
        print(f"using synthetic data at {root} (counts will not match the official split)\n")
    else:
        root = args.data_root or find_cityscapes_root()
        assert root, "could not locate Cityscapes; pass --data-root"
        strict = not args.no_strict_count
        print(f"data root: {root}\n")

    random.seed(0)
    t0_label_table()
    t1_file_count(root, strict)
    t2_pairing(root)
    t3_unique_values(root)
    t4_ignore_fraction(root)
    t5_dimensions(root)
    t6_augmentation_alignment(root)
    t7_post_transform_integrity(root)
    t8_class_frequency(root)

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS)-len(failed)}/{len(RESULTS)} automated checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)
    print("DATASET SUITE PASSED (the overlay inspection in t6 is still manual)")
