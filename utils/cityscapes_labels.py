"""Cityscapes label definitions.

The table below is the standard Cityscapes labelId -> trainId mapping used by
the official benchmark (cityscapesscripts/helpers/labels.py). It is reproduced
here so the pipeline works on Kaggle without internet access.

`verify_against_cityscapesscripts()` re-checks this table against the official
package when it happens to be installed, so the copy can never silently drift.
"""

from collections import namedtuple

Label = namedtuple("Label", ["name", "id", "trainId", "color"])

# name, labelId, trainId, RGB color
LABELS = [
    Label("unlabeled",            0, 255, (0, 0, 0)),
    Label("ego vehicle",          1, 255, (0, 0, 0)),
    Label("rectification border", 2, 255, (0, 0, 0)),
    Label("out of roi",           3, 255, (0, 0, 0)),
    Label("static",               4, 255, (0, 0, 0)),
    Label("dynamic",              5, 255, (111, 74, 0)),
    Label("ground",               6, 255, (81, 0, 81)),
    Label("road",                 7,   0, (128, 64, 128)),
    Label("sidewalk",             8,   1, (244, 35, 232)),
    Label("parking",              9, 255, (250, 170, 160)),
    Label("rail track",          10, 255, (230, 150, 140)),
    Label("building",            11,   2, (70, 70, 70)),
    Label("wall",                12,   3, (102, 102, 156)),
    Label("fence",               13,   4, (190, 153, 153)),
    Label("guard rail",          14, 255, (180, 165, 180)),
    Label("bridge",              15, 255, (150, 100, 100)),
    Label("tunnel",              16, 255, (150, 120, 90)),
    Label("pole",                17,   5, (153, 153, 153)),
    Label("polegroup",           18, 255, (153, 153, 153)),
    Label("traffic light",       19,   6, (250, 170, 30)),
    Label("traffic sign",        20,   7, (220, 220, 0)),
    Label("vegetation",          21,   8, (107, 142, 35)),
    Label("terrain",             22,   9, (152, 251, 152)),
    Label("sky",                 23,  10, (70, 130, 180)),
    Label("person",              24,  11, (220, 20, 60)),
    Label("rider",               25,  12, (255, 0, 0)),
    Label("car",                 26,  13, (0, 0, 142)),
    Label("truck",               27,  14, (0, 0, 70)),
    Label("bus",                 28,  15, (0, 60, 100)),
    Label("caravan",             29, 255, (0, 0, 90)),
    Label("trailer",             30, 255, (0, 0, 110)),
    Label("train",               31,  16, (0, 80, 100)),
    Label("motorcycle",          32,  17, (0, 0, 230)),
    Label("bicycle",             33,  18, (119, 11, 32)),
    Label("license plate",       -1, 255, (0, 0, 142)),
]

NUM_CLASSES = 19
IGNORE_INDEX = 255

# trainId -> name, index 0..18
CLASS_NAMES = [l.name for l in sorted(
    (l for l in LABELS if l.trainId not in (255, -1)), key=lambda l: l.trainId)]

# trainId -> RGB
TRAIN_ID_COLORS = [l.color for l in sorted(
    (l for l in LABELS if l.trainId not in (255, -1)), key=lambda l: l.trainId)]

# Grouping used ONLY for the supplementary thin-vs-large analysis.
# It never changes how official mIoU is computed.
CLASS_GROUPS = {
    "large_region": [0, 1, 2, 8, 9, 10],          # road sidewalk building veg terrain sky
    "medium_object": [3, 4, 13, 14, 15, 16],      # wall fence car truck bus train
    "small_thin": [5, 6, 7, 11, 12, 17, 18],      # pole tlight tsign person rider moto bike
}


def build_id_to_trainid_lut():
    """uint8 lookup table of length 256 mapping labelId -> trainId.

    Any labelId not in the official table maps to 255, which is the safe
    default: an unknown id is ignored rather than guessed.
    """
    import numpy as np
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for l in LABELS:
        if 0 <= l.id < 256:
            lut[l.id] = l.trainId if l.trainId != -1 else IGNORE_INDEX
    return lut


def verify_against_cityscapesscripts():
    """Return (ok, message). Only checks when the official package is present."""
    try:
        from cityscapesscripts.helpers.labels import labels as official
    except Exception as e:  # not installed -> nothing to verify against
        return None, f"cityscapesscripts not available ({type(e).__name__}); using built-in table"

    ours = {l.id: (l.trainId if l.trainId != -1 else 255) for l in LABELS}
    theirs = {l.id: (l.trainId if l.trainId != -1 else 255) for l in official}
    diffs = [(k, ours.get(k), theirs.get(k))
             for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k)]
    if diffs:
        return False, f"MISMATCH vs cityscapesscripts: {diffs}"
    return True, f"built-in table matches cityscapesscripts ({len(theirs)} labels)"


def colorize(mask, ignore_color=(0, 0, 0)):
    """trainId mask [H,W] -> RGB uint8 [H,W,3]. 255 renders as ignore_color."""
    import numpy as np
    mask = np.asarray(mask)
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    out[...] = ignore_color
    for t, c in enumerate(TRAIN_ID_COLORS):
        out[mask == t] = c
    return out


if __name__ == "__main__":
    ok, msg = verify_against_cityscapesscripts()
    print(msg)
    print(f"{NUM_CLASSES} classes:")
    for t, (n, c) in enumerate(zip(CLASS_NAMES, TRAIN_ID_COLORS)):
        print(f"  {t:2d}  {n:<14s} {c}")
