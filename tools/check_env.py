"""Milestone 0 gate: environment, dataset discovery, pretrained weights.

Run this first in any new Kaggle session. It prints a PASS/FAIL line per
requirement and writes results/environment.json for the README.

    python -m tools.check_env
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = []


def check(name, ok, detail="", fix=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok and fix:
        print(f"       fix: {fix}")
    return ok


def main():
    import torch
    check("torch importable", True, torch.__version__)
    check("CUDA available", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU",
          "Kaggle: Settings -> Accelerator -> GPU P100 or T4 x2")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"       {p.name}, {p.total_memory/1024**3:.1f} GB, "
              f"CUDA {torch.version.cuda}, capability {p.major}.{p.minor}")

    try:
        import timm
        from timm.models.layers import trunc_normal_tf_          # noqa: F401
        from timm.models.helpers import named_apply              # noqa: F401
        check("timm with EMCAD's imports", True, getattr(timm, "__version__", "?"))
    except Exception as e:
        check("timm with EMCAD's imports", False, f"{type(e).__name__}: {e}",
              "pip install timm")

    ok_lib = os.path.isdir("lib") and os.path.exists("lib/networks.py")
    check("official EMCAD lib/ present", ok_lib, os.path.abspath("lib") if ok_lib else "",
          "git clone https://github.com/SLDGroup/EMCAD && cp -r EMCAD/lib .")

    if ok_lib:
        try:
            from lib.networks import EMCADNet
            m = EMCADNet(num_classes=19, pretrain=False)
            heads = sum(p.numel() for n, p in m.named_parameters() if "out_head" in n)
            check("EMCADNet builds at num_classes=19", True,
                  f"{sum(p.numel() for p in m.parameters()):,} params, "
                  f"{heads:,} in the four heads")
            del m
        except Exception as e:
            check("EMCADNet builds at num_classes=19", False, f"{type(e).__name__}: {e}")

    # pretrained PVTv2-B2
    cands = (glob.glob("./pretrained_pth/pvt/pvt_v2_b2.pth")
             + glob.glob("/kaggle/input/**/pvt_v2_b2.pth", recursive=True))
    check("PVTv2-B2 pretrained weights", bool(cands),
          cands[0] if cands else "not found",
          "Attach a Kaggle dataset containing pvt_v2_b2.pth, then "
          "mkdir -p pretrained_pth/pvt && cp <path>/pvt_v2_b2.pth pretrained_pth/pvt/ "
          "(or train with --no-pretrain and say so in the README)")
    if cands and not os.path.exists("./pretrained_pth/pvt/pvt_v2_b2.pth"):
        print(f"       copy it: mkdir -p pretrained_pth/pvt && "
              f"cp {cands[0]} pretrained_pth/pvt/")

    # torch.load behaviour (lib/networks.py:68 has no weights_only)
    major_minor = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if major_minor >= (2, 6):
        print("       NOTE: torch >= 2.6 defaults torch.load(weights_only=True); "
              "lib/networks.py:68 may need weights_only=False. "
              "See MANUAL.md section 27.")

    from datasets.cityscapes import find_cityscapes_root
    root = find_cityscapes_root()
    check("Cityscapes located", bool(root), root or "no leftImg8bit/+gtFine/ pair found",
          "Attach a Cityscapes dataset with leftImg8bit/ and gtFine/, or pass --data-root")
    if root:
        n_tr = len(glob.glob(os.path.join(root, "leftImg8bit", "train", "*", "*.png")))
        n_va = len(glob.glob(os.path.join(root, "leftImg8bit", "val", "*", "*.png")))
        n_tid = len(glob.glob(os.path.join(root, "gtFine", "train", "*", "*_labelTrainIds.png")))
        n_lid = len(glob.glob(os.path.join(root, "gtFine", "train", "*", "*_labelIds.png")))
        print(f"       train {n_tr} val {n_va} | trainIds masks {n_tid} | labelIds masks {n_lid}")
        if n_tid == 0 and n_lid > 0:
            print("       labelTrainIds absent -> run tools/make_train_ids.py "
                  "(or csCreateTrainIdLabelImgs)")

    free = None
    try:
        st = os.statvfs("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
        free = st.f_bavail * st.f_frsize / 1024 ** 3
        check("writable output space", free > 3, f"{free:.1f} GB free",
              "Delete old checkpoints; keep only last.pth and best_miou.pth")
    except Exception:
        pass

    from utils.cityscapes_labels import verify_against_cityscapesscripts
    ok, msg = verify_against_cityscapesscripts()
    print(f"[{'PASS' if ok else 'INFO'}] label table -- {msg}")

    os.makedirs("results", exist_ok=True)
    from utils import profiling
    profiling.save_json(
        {"checks": [{"name": n, "ok": o, "detail": d} for n, o, d in CHECKS],
         "cityscapes_root": root, "pretrained_candidates": cands,
         "free_output_gb": free, **profiling.env_info()},
        "results/environment.json")

    failed = [n for n, o, _ in CHECKS if not o]
    print(f"\n{len(CHECKS)-len(failed)}/{len(CHECKS)} checks passed -> results/environment.json")
    if failed:
        print("BLOCKED: " + ", ".join(failed))
        sys.exit(1)
    print("MILESTONE 0 PASSED")


if __name__ == "__main__":
    main()
