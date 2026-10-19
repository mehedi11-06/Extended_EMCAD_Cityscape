# EMCAD → Cityscapes: 19-class semantic segmentation

Adaptation of the official **EMCAD** decoder (Rahman, Munir & Marculescu,
*EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical
Image Segmentation*, CVPR 2024 — [paper](https://arxiv.org/abs/2405.06880),
[code](https://github.com/SLDGroup/EMCAD)) to 19-class semantic segmentation on
Cityscapes, with a reproducible training/evaluation pipeline, efficiency
profiling, and a failure analysis focused on small and thin classes.

The EMCAD architecture in `lib/` is **vendored unmodified** from
SLDGroup/EMCAD @ `<commit hash — results/emcad_commit.txt>`. Verify:

```bash
git clone --depth 1 https://github.com/SLDGroup/EMCAD.git /tmp/EMCAD_ref
diff -r lib /tmp/EMCAD_ref/lib        # expect no output
```

> **Status:** `<fill in: all numbers below come from results/ — replace every
> <placeholder> before submitting>`

---

## 1. Objective

`<2–3 sentences: adapt EMCAD to 19-class Cityscapes, train, evaluate on val,
report per-class IoU / mIoU / peak memory / step time, and analyse failure
modes on thin classes.>`

## 2. Architecture adaptation

EMCAD required **no architectural rewrite** for multiclass segmentation.
`num_classes` enters `EMCADNet.__init__` (`lib/networks.py:11`) and is consumed
at exactly four lines, 83–86, as the output channels of four 1×1 heads:

```python
self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)   # 512 -> 19
self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)   # 320 -> 19
self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)   # 128 -> 19
self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)   #  64 -> 19
```

The PVTv2-B2 encoder, the decoder's feature channels `[512,320,128,64]`, and
every attention module (CAB, SAB, LGAG, MSCB) are independent of the class
count. Measured: the four heads hold **19,532** of **26,782,915** parameters,
so moving from Synapse's 9 classes to 19 changes **10,280 parameters (0.04%)**.

`forward` returns four full-resolution logit maps `[B,19,H,W]`; evaluation uses
`P[-1]` (the finest head), matching the official EMCAD inference path
(`utils/utils.py:268`).

One input constraint: the fixed `scale_factor` upsampling at
`lib/networks.py:107-110` requires H and W divisible by 32. A non-divisible
crop raises a `RuntimeError` inside the decoder rather than failing silently.

The substantive work was the **task pipeline**: dataset and label handling,
ignore-index correctness in both losses and the metric, confusion-matrix
evaluation, profiling and visualization. See `MANUAL.md` §5.

## 3. Cityscapes label handling

- Masks read from `*_gtFine_labelTrainIds.png`; source used in this run:
  `<official | generated with tools/make_train_ids.py>`.
- 19 trainIds (0–18); every other label is **255** and contributes to neither
  loss nor IoU.
- The labelId→trainId table (`utils/cityscapes_labels.py`) cross-checks itself
  against `cityscapesscripts` when installed and refuses to run on disagreement.
- Masks use nearest-neighbour interpolation everywhere; images use bilinear.
  Crop padding uses 255 for masks.

## 4. Environment

| | |
|---|---|
| GPU | `<summary.json: gpu_name>` |
| PyTorch | `<...>` |
| CUDA | `<...>` |
| Driver | `<...>` |
| Platform | `<Kaggle Notebook, GPU accelerator>` |

Install: `pip install -r requirements.txt`, then copy `lib/` from the official
repo (see `kaggle/EMCAD_Cityscapes_Kaggle.ipynb`, cell M0.2).

## 5. Exact commands

```bash
# gates
python -m tools.check_env
python -m tests.test_model_smoke --crop 512 1024 --batch 2 --amp
python -m tests.test_dataset --data-root $ROOT
python -m tests.test_metrics && python -m tests.test_losses
python -m tools.tiny_overfit --data-root $ROOT --images 12 --iters 300 --amp
python -m tools.supervision_cost --crop 512 1024 --batch 2 --amp

# primary experiment
python -W ignore train_cityscapes.py --config configs/cityscapes_pvtv2b2.yaml \
    --data-root $ROOT --out-dir runs/primary

# evaluation, figures, analysis
python -W ignore eval_cityscapes.py --checkpoint runs/primary/best_miou.pth \
    --data-root $ROOT --out-dir results --save-preds results/preds_trainids
python -W ignore visualize_cityscapes.py --checkpoint runs/primary/best_miou.pth \
    --data-root $ROOT --num-images 8 --strategy spread
python -m tools.analyze_failures
```

## 6. Experimental configuration

| Item | Value |
|---|---|
| Encoder | PVTv2-B2, pretrained (`sha256 <results/pretrained_sha256.txt>`) |
| Decoder | EMCAD, unmodified |
| Classes / ignore index | 19 / 255 |
| Train crop | 512×1024 |
| Augmentation | random scale 0.5–2.0, random crop, horizontal flip, colour jitter |
| Validation resolution | `<native 1024×2048 | 512×1024>` |
| Batch size | `<...>` (grad accumulation `<...>`) |
| Optimizer | AdamW, lr `<...>`, weight decay `<...>` |
| Schedule | `<n>` iterations, poly power 0.9, `<n>` warmup |
| Loss | `0.3·CE(ignore_index=255) + 0.7·masked multiclass Dice` |
| Supervision | `<mode>` (`<n>` loss terms) |
| AMP | `<on/off>` |
| Seed | `<...>`, `cudnn.deterministic=True` |

### Deviations from official EMCAD defaults

| Deviation | Reason |
|---|---|
| Supervision `<mode>` instead of `mutation` | `<cite results/supervision_cost.csv>` |
| Masked Dice replaces `utils/utils.py:DiceLoss` | the original counts ignored pixels in its denominator (`MANUAL.md` §9) |
| `CrossEntropyLoss(ignore_index=255)` | the original has no ignore index and raises on 255 |
| Poly LR reinstated | commented out at `trainer.py:118`; standard for Cityscapes |
| AMP enabled | memory headroom at 512×1024 |
| Iteration-based schedule + resume | bounded GPU sessions |
| `<torch.load(weights_only=False) at networks.py:68, if applied>` | `<torch ≥ 2.6 default change>` |

## 7. Results

### Overall

| Metric | Value |
|---|---|
| **mIoU (19 classes)** | `<...>` |
| Pixel accuracy | `<...>` |
| Classes counted | `<...>` / 19 |
| Evaluated head | `P[-1]` (finest) |

### Per-class IoU

`<paste results/per_class_iou.csv as a markdown table — template in MANUAL.md §24>`

### Supplementary grouping (does **not** affect the mIoU above)

| Group | Mean IoU (%) | Mean train pixel share (%) |
|---|---|---|
| Large regions (road, sidewalk, building, vegetation, terrain, sky) | | |
| Medium objects (wall, fence, car, truck, bus, train) | | |
| Small / thin (pole, traffic light, traffic sign, person, rider, motorcycle, bicycle) | | |

### Efficiency

| Measurement | Value |
|---|---|
| Peak GPU memory (allocated) | `<...>` GiB |
| Peak GPU memory (reserved) | `<...>` GiB |
| Mean training step time | `<...>` s ± `<...>` over `<n>` iterations |
| Warm-up excluded | `<n>` iterations |
| Mean dataloading wait (excluded) | `<...>` s |

**Step time is defined as** forward + loss + backward + optimizer.step +
zero_grad, measured with `torch.cuda.synchronize()` immediately before starting
and immediately after finishing the timed GPU work, averaged after a
20-iteration warm-up, excluding dataloading and validation.

## 8. Qualitative results

`figures/qualitative_<scene>.png` — Input RGB | Ground truth | Prediction |
Overlay | Error map. Error map: grey = ignored, white = correct, red =
thin-class false positive, blue = thin-class false negative, orange = other
confusion.

Scenes were selected with `--strategy spread`, which ranks the whole validation
split by per-image mIoU and samples evenly across that range, so the figures are
not only best cases. `figures/qualitative_manifest.json` records the checkpoint,
resolution, strategy and each scene's mIoU.

## 9. Failure analysis

`<Write from results/failure_analysis.json and the panels. Pair every visual
claim with a number. Use hedged language — "may indicate", "suggests", "is
consistent with". Cover: road/sidewalk boundaries, poles, traffic lights,
traffic signs, person/rider confusion, bicycle/motorcycle, distant vehicles,
occlusion. Include the precision/recall distinction for thin classes and the
caveat that IoU penalizes a fixed boundary error far more for thin objects than
large ones, so part of the measured gap is metric geometry rather than model
failure.>`

## 10. Validation of the evaluation code

- `tests/test_metrics.py` — 8 unit tests (perfect, all-wrong, two-class swap,
  all-ignore, mixed ignore, absent classes, hand-computed case, accumulation).
- `tests/test_losses.py` — ignore-invariance, zero gradient at ignored pixels,
  and a numerical demonstration that the original EMCAD Dice responds to
  ignored-region logits while the masked version does not.
- `tests/test_dataset.py` — 8 dataset checks plus manual overlay inspection.
- `logs/overfit.json` — tiny-overfit evidence (12 images, augmentation off,
  pass criterion computed only over classes actually present).
- Official-evaluator cross-check: `<result, or why it was not run>`.

## 11. Reproducibility

Seeds fixed for `random`, `numpy`, `torch`, CUDA and dataloader workers;
`cudnn.deterministic=True`; configuration dumped to `runs/<exp>/config.json`
and embedded in every checkpoint.

Caveats, stated rather than glossed: resuming does not restore dataloader
shuffling or RNG state, so a run completed across several sessions is not
bit-identical to an uninterrupted one; and results are not expected to be
bit-identical across GPU models or PyTorch versions.

## 12. Limitations

`<at least five: short schedule, half-resolution training, single seed,
supervision deviation, evaluation protocol, no multi-scale inference>`

## 13. Repository layout

See `MANUAL.md` §22. `MANUAL.md` is the full technical manual: milestones,
gates, loss derivations, profiling protocol, and the troubleshooting tree.

## Attribution

Architecture and the `lib/` directory: [SLDGroup/EMCAD](https://github.com/SLDGroup/EMCAD),
© its authors, used under its license (see `LICENSE`, `NOTICE`). The training
loop, loss and dataset code here derive from that repository's
`trainer.py`, `utils/utils.py` and `utils/dataset_synapse.py`; each file names
the source lines it derives from.

```bibtex
@inproceedings{rahman2024emcad,
  title={EMCAD: Efficient multi-scale convolutional attention decoding for medical image segmentation},
  author={Rahman, Md Mostafijur and Munir, Mustafa and Marculescu, Radu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11769--11779},
  year={2024}
}
```
