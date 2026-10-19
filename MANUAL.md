# EMCAD → Cityscapes: technical study-and-execution manual

Target: a correct, defensible, reproducible 19-class Cityscapes experiment with
the official EMCAD architecture, plus the report and interview preparation
around it. Organized by **dependency**, not by date.

Every claim about EMCAD below cites a file and line from the official
repository (SLDGroup/EMCAD, CVPR 2024) or shows the code. Every number
attributed to a measurement was produced by code in this repository — nothing
in this manual is a fabricated result. Where a value depends on your GPU, the
manual says where to find it, not what it will be.

---

## 0. Corrections to earlier advice

Stated here first because three things I told you before running the model were
wrong or imprecise.

**0.1 — Non-divisible input does not fail silently.** I said a crop like
769×769 would "silently produce 768×768 outputs." Wrong. The failure happens
inside the decoder, long before the loss, because the EUCB upsample-by-2 and
the encoder skip disagree by one pixel at the additive aggregation
(`lib/decoders.py:355` `d3 = d3 + x3`). Verified by running it:

```
input 769x769 -> RuntimeError: The size of tensor a (50) must match the size of
                 tensor b (49) at non-singleton dimension 3
input 130x260 -> RuntimeError: The size of tensor a (18) must match the size of
                 tensor b (17) at non-singleton dimension 3
```

This is good news: the constraint is enforced loudly, not silently. The
divisible-by-32 rule still holds, and `datasets/cityscapes.py` asserts it at
construction time with that reasoning in the message. `tests/test_model_smoke.py
--check-divisibility` reproduces the error on demand.

**0.2 — Exact parameter counts, measured not estimated.** Running
`EMCADNet(num_classes=19, encoder='pvt_v2_b2')`:

| Component | Parameters |
|---|---|
| PVTv2-B2 encoder | 24,849,856 |
| EMCAD decoder | 1,913,515 |
| Four output heads (19 classes) | 19,532 |
| **Total** | **26,782,915** |

The four heads are `(512+320+128+64)×19 + 4×19 = 19,532` parameters. At 9
classes (Synapse) they would be 9,252. The entire difference between the
Synapse model and the Cityscapes model is **10,280 parameters, 0.04% of the
network.** That single sentence is the strongest possible framing of section 5.

**0.3 — The CE sanity value is right but the tolerance matters.** Measured sum
of four CE terms at initialization on a random batch: **12.1174**, against the
analytic `4·ln(19) = 11.7778`. Close, but not exact, because the heads are not
perfectly uniform at init. `tests/test_model_smoke.py` accepts ±35%, which
catches a wrong label range or a broken head while tolerating normal variance.

**0.4 — The Dice defect, now measured rather than argued.** I claimed the
official `DiceLoss` is unsafe for ignore pixels. `tests/test_losses.py`
demonstrates it numerically: perturbing the logits **only inside the ignored
region**, leaving every valid pixel untouched,

```
official  DiceLoss: 0.908369 -> 0.922949   (delta 0.014580)
masked    DiceLoss: 0.905256 -> 0.905256   (delta 0.000000)
official loss leaks gradient into ignored pixels (max |g| = 7.54e-06)
```

A loss that moves by 0.0146 when nothing a metric would score has changed is a
measurement bug, not a stylistic preference. That test is a deliverable: it is
the evidence for the one loss change you made.

**0.5 — Also confirmed by running it:** `nn.CrossEntropyLoss()` as constructed
at `trainer.py:68` raises `IndexError` on a label of 255. It cannot be used
unchanged on Cityscapes. It fails loudly, so you would find it immediately —
but the "obvious" workaround of clamping 255 to 0 would be far worse than the
Dice bug, because it trains the ego-vehicle region as road.

---

## 1. Final deliverables

### REQUIRED — explicitly asked for

| # | Deliverable | Artifact produced by |
|---|---|---|
| R1 | EMCAD adapted to 19-class Cityscapes | `train_cityscapes.py`, `datasets/cityscapes.py`, `losses/segmentation_losses.py`, `lib/` unmodified |
| R2 | Trained model | `runs/<exp>/best_miou.pth`, `last.pth` (carry config + env inside) |
| R3 | Per-class IoU | `results/per_class_iou.csv` (trainId, name, iou, tp, fp, fn) |
| R4 | mIoU | `results/summary.json` → `miou` |
| R5 | Peak GPU memory | `results/summary.json`, `runs/<exp>/train_summary.json` |
| R6 | Training step time | `train_summary.json` → `mean_step_time_s`, `std`, `timed_iterations`, `warmup_iterations` |
| R7 | Prediction overlays | `figures/qualitative_<scene>.png`, 5-panel |
| R8 | Thin-class failure analysis | README section + `results/failure_analysis.json`, `thin_vs_large.csv`, `figures/iou_vs_frequency.png`, `confusion_heatmap.png` |
| R9 | Reproducible repository | this tree + committed `logs/`, `results/`, `configs/` |
| R10 | Concise research summary | `RESEARCH_SUMMARY.md` (outline in section 25) |

### TECHNICALLY NECESSARY — unstated, but the experiment is invalid without them

| # | Deliverable | Artifact | Why it is not optional |
|---|---|---|---|
| N1 | Masked multiclass Dice | `losses/segmentation_losses.py` | §0.4 |
| N2 | CE with `ignore_index=255` | same file | §0.5 |
| N3 | Confusion-matrix metric | `utils/metrics.py` | per-class IoU at scale |
| N4 | Metric + loss unit tests | `tests/test_metrics.py`, `tests/test_losses.py`, output in `logs/` | R3/R4 are worthless if the metric is wrong |
| N5 | Dataset validation suite | `tests/test_dataset.py`, `figures/sanity_overlays/`, `logs/dataset_validation.log` | silent label bugs are the dominant failure mode |
| N6 | Tiny-overfit evidence | `logs/overfit.json` | cheapest proof the chain is correct |
| N7 | Profiling harness | `utils/profiling.py` | hand-typed numbers are not defensible |
| N8 | Seed + config dump | `configs/*.yaml`, `runs/<exp>/config.json` | reproducibility claim |
| N9 | Training log | `runs/<exp>/train_log.csv` | evidence of stability, not cherry-picking |
| N10 | Which head is evaluated | README + `summary.json` → `prediction_head` | four heads exist; ambiguous otherwise |
| N11 | Documented deviations | README "Architecture adaptation" | supervision mode, AMP, crop, schedule |
| N12 | Class-frequency table | `results/class_frequency.csv` | makes the thin-class analysis quantitative |
| N13 | Environment record | `results/environment.json` from `tools/check_env.py` | GPU, torch, CUDA, driver, paths |

### OPTIONAL — real value, not required

| # | Deliverable | Artifact |
|---|---|---|
| O1 | CE vs CE+Dice ablation | second run dir + README table |
| O2 | Supervision-mode cost | `results/supervision_cost.csv` — cheapest high-value item |
| O3 | Official-evaluator cross-check | `tools/official_eval_crosscheck.py` output |
| O4 | IoU vs frequency scatter | `figures/iou_vs_frequency.png` (already produced by O-tooling) |
| O5 | FLOPs/params | `summary.json` (EMCAD already imports `thop`/`ptflops`) |
| O6 | Inference latency separate from step time | `summary.json` |

O2 and O3 have the best value-per-hour: one turns the supervision decision into
a measurement, the other turns your evaluator from "I think it's right" into
"it agrees with the official tool."

---

## 2. Conceptual study roadmap

Depth tags: **working** = implement from scratch; **solid** = explain precisely
and spot bugs; **orienting** = know what it does and why it is there.

**A. Semantic segmentation fundamentals.** Dense per-pixel classification, no
instances. EMCAD targets medical images where foreground/background dominates;
Cityscapes is 19 mutually exclusive classes in cluttered scenes. The network is
identical, the statistics are not. *working.* Study: EMCAD paper §1, §3 intro.
Answer: why does the Synapse evaluation skip class 0 (`utils/utils.py:272`,
`for i in range(1, classes)`) and why must yours not?

**B. Multiclass logits and target tensors.** Logits `[B,C,H,W]` float,
unnormalized; targets `[B,H,W]` int64 class indices. Every shape bug traces to
one of these. *working.* Study: PyTorch `nn.CrossEntropyLoss` "Shape" section;
`trainer.py:85`. Answer: what does `label_batch.squeeze(1)` assume, and why does
`datasets/cityscapes.py` make it unnecessary?

**C. CrossEntropyLoss.** Fuses `log_softmax` + NLL; must receive raw logits.
`ignore_index` removes pixels from numerator and denominator. *working.* Study:
`trainer.py:68,111`; `tests/test_losses.py::test_ce_rejects_255_without_ignore_index`.
Answer: if you softmax before CE, does it raise? (No — it silently distorts.)

**D. Dice loss.** `2|P∩G|/(|P|+|G|)`, differentiable via soft probabilities;
counteracts CE's domination by frequent classes. *working.* Study:
`utils/utils.py:102-138` line by line; §9 below. Answer: what are `y_sum` and
`z_sum`, and why does the squared form differ from standard Dice?

**E. Focal loss.** CE reweighted by `(1-p_t)^γ`. Offered by the professor as an
alternative; you are not required to use it. Know it well enough to justify
choosing CE+Dice. *orienting.* Study: Lin et al. §3. Answer: what does γ=0
reduce to?

**F. Cityscapes IDs vs trainIds.** 34 `labelIds`; 19 evaluation classes mapped
to 0–18, everything else 255. Training on `labelIds` runs fine and means
nothing. *working.* Study: `utils/cityscapes_labels.py` (the full table, with
`verify_against_cityscapesscripts()`). Answer: which labelIds have
`ignoreInEval=True`, and what trainId does the ego vehicle get?

**G. `ignore_index=255`.** A contract spanning dataloader, **both** losses, and
the metric — CE's `ignore_index` does nothing for your Dice or your confusion
matrix. *working.* Study: §9; `utils/metrics.py`. Answer: what fraction of a
Cityscapes frame is ignored, and where is it concentrated? (Measured by
`tests/test_dataset.py` t4 — read your own number.)

**H. Confusion matrices.** 19×19, rows = GT, columns = prediction, built with
`np.bincount(19*gt + pred)` over valid pixels. *working.* Study:
`utils/metrics.py`. Answer: which cell/row/column give TP, FP, FN for class c?

**I. IoU and mIoU.** `IoU_c = M[c,c] / (row_c + col_c − M[c,c])`; mIoU is the
**unweighted** mean, so one failed thin class costs as much as one failed large
class. *working.* Answer: what does your formula return for a class absent from
the entire split, and why does that matter for the mean?

**J. Augmentation for segmentation.** Random scale 0.5–2.0, random crop,
horizontal flip, photometric jitter. All geometric ops synchronized by one
random draw. *working.* Study: `dataset_synapse.py:12-46` as a structural
template; `datasets/cityscapes.py::_joint_transform` as the Cityscapes version.
Answer: why are vertical flip and ±20° rotation (which the repo does use) wrong
for street scenes?

**K. Mask interpolation.** Bilinear on a label map invents labels: averaging 5
(pole) and 2 (building) gives 3.5 → class 3 (wall). Worse at ignore boundaries,
where interpolating 255 against 0 can manufacture any class. *working.* Study:
`dataset_synapse.py:40-41` (`order=3` image, `order=0` label). Answer: this is
exactly why `tests/test_dataset.py` t7 works as a detector — explain the
mechanism.

**L. Encoder–decoder networks.** EMCAD is a *decoder* paper. *solid.* Study:
paper Fig. 2 with `lib/decoders.py:341-389`. Answer: at what stride is the
finest decoder feature, and what does that imply for the thinnest structure
representable before upsampling?

**M. PVTv2-B2.** Four stages at strides 4/8/16/32, `embed_dims=[64,128,320,512]`,
`depths=[3,4,6,3]`, `sr_ratios=[8,4,2,1]` (`lib/pvtv2.py:404-409`). Overlapping
patch embedding and conv-in-FFN rather than fixed positional embeddings, so
variable non-square input works — confirmed by running 512×1024 and 128×256.
*solid.* Answer: where are H and W computed per stage, and why does the stored
`img_size=224` not constrain you?

**N. EMCAD decoder.** Per stage: MSCAM (CAB → SAB → MSCB) → EUCB → LGAG →
additive aggregation. Returns `[d4,d3,d2,d1]` at strides 32/16/8/4 with channels
`[512,320,128,64]`. *working.* Study: `lib/decoders.py:315-389`. Answer: the
order inside one MSCAM, and whether attention is multiplicative or additive.

**O. Multi-scale depthwise convolution (MSDC/MSCB).** Parallel depthwise convs
with kernels 1/3/5 over an expanded channel space, summed or concatenated, then
channel shuffle and pointwise projection (`decoders.py:77-175`). EMCAD's
efficiency claim, and directly relevant to thin classes. *solid.* Answer: with
`expansion_factor=2`, `add=True`, how many channels enter `pconv2`?

**P. Channel attention (CAB).** Shared MLP over avg- and max-pooled
descriptors, summed, sigmoid, broadcast spatially (`decoders.py:252-290`),
ratio 16 clamped when `in_channels < ratio`. *solid.* Answer: for
`channels[3]=64`, what is `reduced_channels`, and why is CAB blind to *where*?

**Q. Spatial attention (SAB).** Channel-mean and channel-max → 2 maps → one 7×7
conv → sigmoid (`decoders.py:292-313`). *solid.* Answer: `self.sab` is a single
shared module used at all four stages (`decoders.py:338`) — what does sharing
across resolutions imply?

**R. Large-kernel grouped attention gate (LGAG).** Two grouped k×k convs to
`F_int = F_l//2`, added, activated, 1×1 → one channel → sigmoid → multiplied
into the skip (`decoders.py:219-250`). *solid.* Answer: the gate is a **single**
spatial map, not per-channel — what can it therefore not suppress?

**S. Skip connections.** Gated skips aggregated **additively** (`d3 = d3 + x3`),
so decoder channels equal encoder channels throughout. *solid.* Answer: why does
that force EUCB to project to exactly `channels[i+1]`?

**T. Deep supervision.** Loss on intermediate decoder outputs. Three variants in
`trainer.py:91-113`. *working.* See §6.

**U. Multi-output segmentation heads.** Four 1×1 convs, one per decoder stage,
all upsampled to input resolution (`networks.py:83-110`). *working.* Answer:
which element does inference use, and where is that decided?
(`utils/utils.py:268`, `outputs = P[-1]`.)

**V. Class imbalance.** Cityscapes pixel frequency spans orders of magnitude.
CE optimizes mean pixel likelihood; mIoU weights all 19 equally. That mismatch
is the central tension in your results. *working.* Study: your own
`results/class_frequency.csv`. Answer: what mIoU would a road-only predictor get?

**W. Small/thin-object segmentation.** Poles, lights, signs are few pixels and
mostly boundary: little loss signal, and IoU collapses on a one- or two-pixel
error. At stride 4 a distant pole can be sub-pixel. *working.* Answer: for a
6-pixel-wide pole, what IoU results from a uniform 2-pixel dilation?

**X. CUDA asynchronous execution.** Kernel launches return immediately; a CPU
timer without `torch.cuda.synchronize()` measures launch overhead. *working.*
Study: `utils/profiling.py::StepTimer`. Answer: why does `loss.item()` happen to
hide this, and why is relying on that still wrong?

**Y. GPU peak-memory measurement.** `reset_peak_memory_stats()` then
`max_memory_allocated()` (tensors) or `max_memory_reserved()` (allocator).
Report which. *working.* Answer: why is `nvidia-smi` the wrong instrument?

**Z. Reproducibility.** Seeds for `random`/`numpy`/`torch`/`cuda`,
`cudnn.deterministic`, `worker_init_fn`, config files, committed logs, pinned
versions. *working.* Study: `train_synapse.py:66-76`, `trainer.py:55-56`, and
`train_cityscapes.py::set_seed`. Answer: with a fixed seed and
`deterministic=True`, will two runs give bit-identical mIoU? What else must hold?

---

## 3. EMCAD codebase map

Verified inventory of the clone (images and `lists/` omitted):

```
lib/decoders.py  lib/networks.py  lib/pvtv2.py  lib/resnet.py
utils/utils.py   utils/dataset_synapse.py  utils/dataset_ACDC.py
utils/dataloader.py  utils/dataloader_polyp.py  utils/joint_transforms.py
utils/transforms.py  utils/misc.py  utils/format_conversion.py
utils/preprocess_synapse_data.py  utils/preprocess_synapse_data_3d.py
train_synapse.py  test_synapse.py  train_polyp.py  test_polyp.py  trainer.py
requirements.txt
```

### MODEL — `lib/networks.py` (126 lines)
`class EMCADNet` (`__init__` 11–86, `forward` 88–115). The only model entry
point.
**READ: fully. MODIFY: no**, except possibly `torch.load` at line 68
(`weights_only` under torch ≥ 2.6; the Kaggle notebook patches this
automatically). **COPY: no.**
Questions: where is `num_classes` used, and how many times? What do lines 15–19
and 91–92 do with 3-channel input? Why do `channels` differ between the ResNet
and PVT branches? What constrains H and W (lines 107–110)?

### DECODER — `lib/decoders.py` (389 lines)
`MSDC` (77), `MSCB` (108), `MSCBLayer` (177), `EUCB` (192), `LGAG` (219),
`CAB` (252), `SAB` (292), `EMCAD` (315).
**READ: yes** — `EMCAD.forward` (341–389) carefully. **MODIFY: no.** Leaving
this untouched is itself a reportable result. **COPY: no.**
Questions: write the shape after each of the 16 operations in `forward`. Why is
`self.sab` shared but `cab1..4` separate? What does `channel_shuffle` do after
the `add` reduction (line 168)? What are `F_g/F_l/F_int` at stage 1?

### ENCODER — `lib/pvtv2.py`
`OverlapPatchEmbed` (156), `PyramidVisionTransformerImpr` (196),
`forward_features` (313), `pvt_v2_b2` (404).
**READ: lines 156–163, 196–215, 313–350, 404–409. MODIFY: no. COPY: no.**
Questions: `embed_dims` and `sr_ratios` for B2? Where are H and W computed per
stage? What is the memory consequence of `sr_ratios[3]=1` at stride 32 on a
512×1024 input?
`lib/resnet.py` — **READ: no.** Unused here.

### TRAINING — `train_synapse.py` (128), `trainer.py` (159)
`train_synapse.py`: parser 13–63, snapshot path 104–115, model build 120,
dispatch 126–127. **READ: yes. MODIFY: no. COPY: yes** → `train_cityscapes.py`
(seeding block 66–76; argument names).
`trainer.py`: `inference` 22–38, `trainer_synapse` 40–159, loss block 102–113,
optimizer 72, disabled poly LR 118–119. **READ: closely. MODIFY: no. COPY: yes,
corrected.**
Questions: print `ss` under each supervision mode. Why is `iout` a *sum of
logits*? Why is the LR schedule commented out? How often does Synapse validate,
and is that affordable on 500 Cityscapes images?

### LOSS — `utils/utils.py:102-138` (`DiceLoss`), `:19-28` (`powerset`)
**READ: line by line. MODIFY: no. COPY: yes, with the ignore fix** →
`losses/segmentation_losses.py`.
Questions: what does `_one_hot_encoder` give for a 255 pixel? Which term still
sees it? Linear or squared denominator? Batch-level or per-image?

### DATASET — `utils/dataset_synapse.py`
`random_rot_flip` (12), `random_rotate` (22), `RandomGenerator` (29),
`Synapse_dataset` (49). **READ: as a contract reference. MODIFY: no. COPY: the
contract only.**
Questions: what keys/shapes/dtypes does `__getitem__` return? Which
interpolation order for image (40) vs label (41)? Why `.float()` then `.long()`?
Which augmentations are physically meaningless for street scenes?
`dataloader.py`, `dataloader_polyp.py`, `dataset_ACDC.py`, `transforms.py` —
skip. `joint_transforms.py` — skim for the synchronization pattern.

### EVALUATION — `utils/utils.py:239-274` (`val_single_volume`)
Lines 252–255 are decisive: `P = net(input)`, `outputs = P[-1]`,
`argmax(softmax(outputs))`. **READ: yes. MODIFY: no. COPY: the `P[-1]`
convention** → `eval_cityscapes.py`.
Questions: which head is evaluated? Why `range(1, classes)` and why must yours
start at 0?
`utils/utils.py:140+` (`calculate_metric_percase`, medpy Dice/HD95) — **READ:
no.** `test_synapse.py` — skim for the checkpoint-loading pattern.

### UTILITIES — `utils/utils.py:66-84` (`CalParams`, thop/ptflops)
**READ: skim**; useful only for O5. Note the heavy module-level imports
(`medpy`, `SimpleITK`, `seaborn`, `segmentation_mask_overlay`, `thop`,
`ptflops`): importing `utils.utils` pulls all of them, several of which are not
installed on Kaggle. **Never import it wholesale** — this repo deliberately does
not.
`preprocess_synapse_data*.py`, `format_conversion.py`, `misc.py` — **READ: no.**

### CONFIGURATION
The repo has none: argparse plus a string-built snapshot path
(`train_synapse.py:104-115`). **MODIFY: no. ADD:** `configs/*.yaml`, copied
into each run directory as `config.json`. **`requirements.txt`: READ and
extend**, pinning what you actually ran.

---

## 4. Complete forward-pass trace

For `[B, 3, 512, 1024]`, `encoder='pvt_v2_b2'`, `num_classes=19`. Encoder and
decoder shapes below were verified by running the model (`lib/` unmodified) and
printing them; the 512×1024 column scales the measured 128×256 run by 4.

**Entry** (`networks.py:88-92`). `x.size()[1] == 3`, so the grayscale branch is
skipped and `self.conv` (lines 15–19) stays unused — untrained dead weight,
worth noting as something you checked rather than removed.

**Encoder** (`networks.py:95` → `pvtv2.py:313-350`):

| Output | Stride | Symbolic | 512×1024 | measured at 128×256 |
|---|---|---|---|---|
| `x1` | 4 | `[B,64,H/4,W/4]` | `[B,64,128,256]` | `(1,64,32,64)` |
| `x2` | 8 | `[B,128,H/8,W/8]` | `[B,128,64,128]` | `(1,128,16,32)` |
| `x3` | 16 | `[B,320,H/16,W/16]` | `[B,320,32,64]` | `(1,320,8,16)` |
| `x4` | 32 | `[B,512,H/32,W/32]` | `[B,512,16,32]` | `(1,512,4,8)` |

Channels are `embed_dims=[64,128,320,512]` (`pvtv2.py:407`).

**Decoder call** (`networks.py:99`): `self.decoder(x4, [x3, x2, x1])`. The
decoder's `channels=[512,320,128,64]` (`networks.py:33`) is `embed_dims`
reversed, which is why additive aggregation needs no projection.

| Step | Operation | Out (512×1024) |
|---|---|---|
| MSCAM4 | `cab4(x)*x` → `sab(·)*·` → `mscb4` | `d4 [B,512,16,32]` |
| EUCB3 | upsample ×2, DW conv, shuffle, PW 512→320 | `[B,320,32,64]` |
| LGAG3 | gate `skips[0]=x3`; `F_int=160`, groups 160 | `[B,320,32,64]` |
| Add3 + MSCAM3 | `d3 = d3 + x3`; `cab3`→`sab`→`mscb3` | `d3 [B,320,32,64]` |
| EUCB2 → LGAG2 → Add2 → MSCAM2 | PW 320→128; `F_int=64` | `d2 [B,128,64,128]` |
| EUCB1 → LGAG1 → Add1 → MSCAM1 | PW 128→64; `F_int=32` | `d1 [B,64,128,256]` |

`EMCAD.forward` returns `[d4, d3, d2, d1]`, coarsest first
(`decoders.py:389`) — verified: `[(1,512,4,8),(1,320,8,16),(1,128,16,32),(1,64,32,64)]`.

**Heads** (`networks.py:83-86,102-105`):

| Head | Conv | Input | Output |
|---|---|---|---|
| `out_head4` | `Conv2d(512,19,1)` | `d4` | `p4 [B,19,16,32]` |
| `out_head3` | `Conv2d(320,19,1)` | `d3` | `p3 [B,19,32,64]` |
| `out_head2` | `Conv2d(128,19,1)` | `d2` | `p2 [B,19,64,128]` |
| `out_head1` | `Conv2d(64,19,1)` | `d1` | `p1 [B,19,128,256]` |

**Upsampling** (`networks.py:107-110`): fixed `scale_factor` 32/16/8/4,
`mode='bilinear'`, no `align_corners` (so `False`), no `size=`. All four become
`[B,19,512,1024]` — verified: all four printed `(1,19,128,256)` for a 128×256
input.

`forward` returns `[p4,p3,p2,p1]`; both the `mode=='test'` branch and the
fall-through return the same list (lines 112–115). Raw logits, no normalization.

**Why all four heads become 19-class.** `num_classes` is used in exactly four
places — lines 83, 84, 85, 86 — as `out_channels`. Nothing else in the model
references it.

| Component | Changes with `num_classes=19`? |
|---|---|
| Encoder stages, embed dims, attention heads | No |
| Decoder feature channels `[512,320,128,64]` | No |
| CAB reduction, SAB kernel, LGAG `F_int`/groups | No |
| MSCB expansion channels, kernel sizes | No |
| **Output projections `out_head1..4`** | **Yes — only these (+10,280 params vs 9 classes)** |
| Upsampling factors | No |

Internal attention channels derive from `channels`, which comes from the
encoder, not from `num_classes`.

**Memory arithmetic that drives §6.** One `[2,19,512,1024]` float32 tensor is
`2·19·524288·4 B ≈ 80 MB`. Four heads at full resolution ≈ 320 MB of logits
before the loss allocates anything.

---

## 5. Verifying "no architectural rewrite"

**Where `num_classes` enters** (`networks.py:11`):

```python
def __init__(self, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2,
             dw_parallel=True, add=True, lgag_ks=3, activation='relu',
             encoder='pvt_v2_b2', pretrain=True, pretrained_dir='./pretrained_pth/pvt/'):
```

Grep the file: `num_classes` appears on lines 11, 83, 84, 85, 86 and nowhere
else. It is not passed to the encoder, not passed to `EMCAD(...)` at line 78,
not referenced in `forward`.

**The four heads** (`networks.py:83-86`, verbatim):

```python
self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
```

**The decoder is class-agnostic.** `networks.py:78` constructs
`EMCAD(channels=..., kernel_sizes=..., expansion_factor=..., dw_parallel=...,
add=..., lgag_ks=..., activation=...)`. No class-count argument exists in
`EMCAD.__init__` (`decoders.py:316`).

**What `forward()` returns**: `[p4,p3,p2,p1]`, raw logits, identical for both
`mode` values (lines 112–115).

**The repo's own usage confirms it.** `train_synapse.py:23-24` declares
`--num_classes` default 9 and line 120 passes it straight through. The same
constructor serves Synapse's 9 classes and the polyp binary case. Nineteen is
intended usage, not a workaround.

**Measured**: the heads hold 19,532 of 26,782,915 parameters. Switching from 9
to 19 classes changes 10,280 parameters, 0.04% of the model.

**Honest caveat.** "No rewrite" is a claim about the class count, not about
everything. The one real constraint is divisibility by 32 (§0.1), satisfied by
choosing the crop, not by editing the model.

### Architecture changes vs task-pipeline changes

**Architecture: one configuration value.** Zero edits to `lib/`.

**Task pipeline: everything else.**

| Area | Weight | What it involved |
|---|---|---|
| Dataset + label handling | Heavy | new `Dataset`, trainIds, joint transforms, nearest-only masks, 255 preserved end to end |
| Loss | Heavy | `ignore_index=255`; masked Dice written from scratch |
| Training supervision | Moderate, a judgement call | cost trade-off that did not exist at 224×224×9 |
| Evaluation | Heavy | confusion-matrix IoU replacing medical Dice/HD95; all 19 classes |
| Profiling | Moderate | no equivalent exists in the repo |
| Visualization | Moderate | Cityscapes palette, 5-panel, error maps |
| Reproducibility | Light | seeding pattern exists; add configs and logs |

The sentence to have ready: *the architectural adaptation reduced to a single
constructor argument, which I verified by tracing where `num_classes` is
consumed and by measuring that it changes 0.04% of the parameters; the
substantive work was the task pipeline — dataset semantics, ignore-index
correctness across three independent components, metric construction, and the
supervision cost trade-off at Cityscapes resolution.* Finding that a
well-designed codebase already generalizes, and **proving** it, is what the
professor's "do not assume EMCAD must be rewritten" instruction was testing.

---

## 6. The three supervision modes

Selected by `--supervision` (`train_synapse.py:44-45`, default `mutation`),
realized in `trainer.py:91-113`:

```python
n_outs = len(P)                        # 4
out_idxs = list(np.arange(n_outs))     # [0,1,2,3]
if args.supervision == 'mutation':
    ss = [x for x in powerset(out_idxs)]
elif args.supervision == 'deep_supervision':
    ss = [[x] for x in out_idxs]
else:
    ss = [[-1]]
```

then, per subset:

```python
iout = 0.0
for idx in range(len(s)):
    iout += P[s[idx]]                  # sum of LOGITS at full resolution
loss_ce   = ce_loss(iout, label_batch[:].long())
loss_dice = dice_loss(iout, label_batch, softmax=True)
loss += (0.3 * loss_ce + 0.7 * loss_dice)
```

**mutation** — `powerset` (`utils/utils.py:19-28`) yields all 16 subsets; the
empty one is skipped at lines 107–108, leaving **15 loss terms**. Each sums its
subset's logits before the loss, so the model is supervised on every additive
combination of heads. Conceptually an implicit ensemble regularizer: no single
head can free-ride. It is EMCAD's default and the setting behind its published
Synapse numbers. `supervision_subsets("mutation", 4)` in this repo reproduces
exactly 15 subsets (asserted in `tests/test_losses.py`).

**deep_supervision** — `[[0],[1],[2],[3]]`, **4 terms**, no logit combination.

**last_layer** — `[[-1]]`, **1 term** on `P[-1] = p1`, the finest head. Note
this matches what `val_single_volume` evaluates (`utils/utils.py:268`), so
train and eval are exactly consistent in this mode.

| | Terms | Heads | Logits combined | Compute | Memory driver | Purpose |
|---|---|---|---|---|---|---|
| mutation | 15 | all 4, all subsets | yes, summed | ~15× loss block | 15 summed tensors + 15 softmax copies retained for backward | implicit multi-scale ensembling |
| deep_supervision | 4 | all 4 separately | no | ~4× | 4 softmax copies | direct gradient to each stage |
| last_layer | 1 | `p1` | n/a | 1× | 1 softmax copy | simplest; exactly matches inference |

### 9×224×224 → 19×512×1024

| | Synapse | Cityscapes |
|---|---|---|
| Classes | 9 | 19 |
| Pixels | 50,176 | 524,288 |
| Elements per logit tensor per image | 451,584 | 9,961,472 |
| Ratio | 1× | **≈22×** |

At batch 2, each `[2,19,512,1024]` tensor is ≈80 MB. Under mutation, each of the
15 subsets materializes a summed logit tensor (~80 MB) plus a softmax tensor
inside Dice (~80 MB), both retained for backward: **on the order of 2.4 GB of
loss-block activations**, on top of encoder/decoder activations and the four
base logits. Deep supervision: ~320 MB. Last layer: ~80 MB. Compute scales
similarly — every term runs a full-resolution softmax over ten million
elements.

A numerical consideration independent of cost: `iout = p4 + p3` sums logits
across scales with equal weight. With 9 balanced organs that is benign. With 19
imbalanced classes the argmax of a sum is dominated by whichever head is more
confident, generally the coarse one — the head that cannot resolve poles. State
that as a hypothesis, not a finding.

### Recommendation

| Criterion | Best |
|---|---|
| Correctness (train/eval consistency) | `last_layer` — supervises exactly the evaluated head |
| Faithfulness to EMCAD | `mutation` — the published default |
| GPU cost | `last_layer` ≫ `deep_supervision` ≫ `mutation` |
| Experiment simplicity | `last_layer` |

**Recommended: `deep_supervision` for the primary run**, `last_layer` as the
OOM fallback, `mutation` measured but not necessarily trained. Deep supervision
keeps EMCAD's multi-output design meaningful — all four heads train, which is
the architectural claim under test — while removing the combinatorial term
whose cost grew 22×. Pure `last_layer` leaves `out_head2/3/4` untrained, which
invites the fair criticism that you did not evaluate EMCAD's multi-scale
decoding at all.

**Do not decide from this reasoning.** Run `python -m tools.supervision_cost
--crop 512 1024 --batch 2 --amp`. It writes `results/supervision_cost.csv` with
measured peak memory and step time for all three, and reports OOM rather than
crashing. If mutation fits at a reasonable step time, use mutation and the
question disappears.

### Will this be questioned in the interview?

Yes, and expect it from this advisor specifically — mutation supervision traces
back through his CASCADE and MERIT work. Treat it as the most likely technical
challenge.

Four-part answer: **state the deviation** ("I trained with deep supervision
rather than EMCAD's default mutation supervision"); **give the measured
reason** (15 terms on full-resolution 19-channel logits; cite your own numbers
from `supervision_cost.csv`); **scope the claim** ("my mIoU is a correctly
implemented EMCAD-on-Cityscapes baseline, not EMCAD's best achievable
Cityscapes performance"); **show the path back** ("the follow-up is mutation at
a reduced crop or with gradient accumulation, everything else fixed").

Indefensible: silently changing the default and reporting the number as EMCAD's.
The deviation is a reasonable engineering decision under a compute constraint;
concealing it is not.

---

## 7. Cityscapes data pipeline

### Directory layout

```
cityscapes/
  leftImg8bit/{train,val,test}/<city>/<city>_<seq>_<frame>_leftImg8bit.png
  gtFine/{train,val,test}/<city>/<city>_<seq>_<frame>_gtFine_{color,instanceIds,labelIds,labelTrainIds}.png
```

Official counts: train 2975, val 500, test 1525 (labels withheld). Images are
1024×2048.

### Filename matching

Pure string substitution, so it can be asserted rather than trusted
(`datasets/cityscapes.py::mask_path_for`):

```
leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png
```

Glob the images, derive each mask path, assert existence. Never glob both
directories and zip them — that misaligns silently if one file is missing.

### labelIds vs labelTrainIds

`labelIds` holds 0–33 under the full taxonomy including non-benchmark classes
(ego vehicle, rectification border, out-of-roi, static, dynamic, ground,
parking, rail track, guard rail, bridge, tunnel, polegroup, caravan, trailer,
license plate). `labelTrainIds` is the official remap: 19 classes to 0–18,
everything else to 255.

### The 19 classes

| trainId | Name | Group (analysis only) | RGB |
|---|---|---|---|
| 0 | road | large | 128,64,128 |
| 1 | sidewalk | large | 244,35,232 |
| 2 | building | large | 70,70,70 |
| 3 | wall | medium | 102,102,156 |
| 4 | fence | medium | 190,153,153 |
| 5 | pole | small/thin | 153,153,153 |
| 6 | traffic light | small/thin | 250,170,30 |
| 7 | traffic sign | small/thin | 220,220,0 |
| 8 | vegetation | large | 107,142,35 |
| 9 | terrain | large | 152,251,152 |
| 10 | sky | large | 70,130,180 |
| 11 | person | small/thin | 220,20,60 |
| 12 | rider | small/thin | 255,0,0 |
| 13 | car | medium | 0,0,142 |
| 14 | truck | medium | 0,0,70 |
| 15 | bus | medium | 0,60,100 |
| 16 | train | medium | 0,80,100 |
| 17 | motorcycle | small/thin | 0,0,230 |
| 18 | bicycle | small/thin | 119,11,32 |

Ignore (255) renders black. These are the standard palette colours; anyone who
works with Cityscapes reads your figures at a glance because of them. The
grouping column is for the supplementary analysis in §20 and never touches
official mIoU.

### Generate trainIds, or consume them?

**Consume if present.** Check:

```bash
find $ROOT/gtFine/train -name "*_labelTrainIds.png" | wc -l   # expect 2975
find $ROOT/gtFine/val   -name "*_labelTrainIds.png" | wc -l   # expect 500
```

`tools/check_env.py` does this and tells you which you have. If absent — common
on Kaggle mirrors — generate them, preferring the official tool when internet
is on:

```bash
pip install cityscapesscripts
CITYSCAPES_DATASET=$ROOT csCreateTrainIdLabelImgs
```

With internet off, `python -m tools.make_train_ids --data-root $ROOT --out-root
/kaggle/working/cityscapes_trainids --link-images` applies the identical table
from `utils/cityscapes_labels.py`, which cross-checks itself against
`cityscapesscripts` whenever that package is importable and refuses to run if
they disagree. It writes to a new root because `/kaggle/input` is read-only.
Never hand-write a 34→19 dictionary.

### Dataset class contract

| Key | Shape | dtype | Allowed values |
|---|---|---|---|
| `image` | `[3,512,1024]` | `float32` | ImageNet-normalized, ≈[-2.5, 2.5] |
| `label` | `[512,1024]` | `int64` | `{0..18} ∪ {255}` only |
| `name` | — | `str` | `frankfurt_000000_000294` |

After collation: `[B,3,512,1024]` and `[B,512,1024]`. The label has **no**
channel dimension, so `trainer.py:85`'s `.squeeze(1)` is not needed — dropped
deliberately rather than carried over.

Validation: no augmentation. Native 1024×2048 is the standard protocol and
avoids resize artifacts in the metric; it is usually affordable at batch 1
because there is no autograd graph. Try it before settling for
`--val-resize 512 1024`, and state the choice in the README either way.

### Joint transforms

Training order (`datasets/cityscapes.py::_joint_transform`):

1. **Random scale** ∈ [0.5, 2.0] on the full frame. Image bilinear, mask nearest.
2. **Pad if smaller than the crop** — image with ImageNet mean, **mask with
   255**, never 0. Padding a mask with 0 invents road.
3. **Random crop** to 512×1024, same box for both.
4. **Horizontal flip** p=0.5, same decision for both.
5. **Photometric jitter** — image only.
6. **Normalize** with ImageNet mean/std (the PVTv2 weights were pretrained with
   it) — image only.
7. **Mask → int64**, no normalization, no ×255.

Must be synchronized: scale, pad, crop box, flip. The guarantee comes from
drawing the random parameters once inside a single function that holds both
tensors — the same structural trick as `RandomGenerator.__call__`
(`dataset_synapse.py:33-41`). Must not touch the mask: normalization, jitter,
blur, float casts.

### Why bilinear for images and nearest for masks

Image values sample a continuous signal, so interpolation is meaningful and
bilinear reduces aliasing. Label values are categorical indices where the
arithmetic is meaningless: averaging 5 (pole) and 2 (building) gives 3.5 → 3
(wall), a class that was nowhere near that pixel. It is worse at ignore
boundaries, where interpolating 255 against 0 can manufacture any class along
every ignore edge — and Cityscapes has an ignore edge around every frame.
Nearest preserves the value set exactly, which is also what makes the
post-transform assertion in §8 a working detector.

---

## 8. Dataset validation suite

Implemented in `tests/test_dataset.py`; run it and keep the output:

```bash
python -m tests.test_dataset --data-root $ROOT 2>&1 | tee logs/dataset_validation.log
```

Each test prints PASS/FAIL plus, on failure, how to interpret it.

**T0 — label table cross-check.** Purpose: the built-in 34-row table must match
`cityscapesscripts` when that package is present. Pass: identical mapping for
every labelId. Failure: do not train; the remap is wrong.

**T1 — file count.** Pass: 2975 train / 500 val. Failure: a short count is a
truncated mirror (re-download — a partial train set silently changes your
result); a long count usually means `test/` was globbed, or `_color.png` matched.

**T2 — pairing.** Every image resolves to a `labelTrainIds` or `labelIds` mask.
Pass: zero unpaired. Failure: trainIds never generated, or the substitution rule
does not fit this mirror's layout — print one derived path beside one real path.

**T3 — raw unique values.** Pass: `⊆ {0..18, 255}` for trainIds files. Failure:
values 19–33 mean you are reading `labelIds`; values >255 or a float dtype mean
you opened the colour PNG.

**T4 — ignore fraction.** Pass: a few percent to low double digits, concentrated
bottom-centre and at the border. Failure: ≈0% means it is not a trainIds map;
>50% means the wrong file or misalignment.

**T5 — dimensions.** Pass: image and mask the same size for every sampled pair.
Failure: a mismatch on a specific frame means the pairing matched the wrong city
or frame id.

**T6 — augmentation alignment.** The highest-value test. Writes
`figures/sanity_overlays/<name>_overlay.png` (RGB | colorized mask | blend) and
asserts shapes and that two draws of the same index differ. **The decisive step
is manual**: open them. Mask boundaries must sit on object boundaries. A
constant offset = crop desync; a mirrored mask = flip applied to one side only;
a scaled mask = separate resize calls. None of these show up in the loss curve.

**T7 — post-transform integrity.** Pass: after full augmentation, `dtype ==
int64` and unique values still `⊆ {0..18, 255}` across 200 samples. Failure: any
value in 19–254 proves bilinear interpolation touched the mask somewhere in the
chain.

**T8 — class frequency.** Writes `results/class_frequency.csv`. Pass: all 19
classes non-zero and a plausible ordering. Failure: a zero-count class means too
small a sample or a remap bug. Keep these fractions — the IoU-vs-frequency
relationship in §20 is built on them.

Sanity note: this suite was exercised end to end against a synthetic Cityscapes
tree (`python -m tests.test_dataset --synthetic`), 10/10 automated checks
passing, before being handed to you. Running `--synthetic` is also a useful way
to verify the code in a Kaggle session before the real dataset is attached.

---

## 9. Loss design

### Why the official DiceLoss is unsafe here

`utils/utils.py:107-138`:

```python
def _one_hot_encoder(self, input_tensor):
    tensor_list = []
    for i in range(self.n_classes):
        temp_prob = input_tensor == i
        tensor_list.append(temp_prob.unsqueeze(1))
    return torch.cat(tensor_list, dim=1).float()

def _dice_loss(self, score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1 - loss
```

Take pixel `j` with label 255 and softmax probabilities `p_{c,j}`, where
`Σ_c p_{c,j} = 1`.

`input_tensor == i` is false for every `i ∈ 0..18`, so the one-hot row is all
zeros: `g_{c,j} = 0`. Then, per class `c`:

- `intersect = Σ_j p_{c,j}·g_{c,j}` — pixel j contributes `p·0 = 0`. Fine.
- `y_sum = Σ_j g_{c,j}²` — contributes 0. Fine.
- `z_sum = Σ_j p_{c,j}²` — **contributes `p_{c,j}² > 0`.**

So the loss is

```
1 − (2·Σ_valid p·g + ε) / (Σ_valid p² + Σ_ignored p² + Σ_valid g² + ε)
```

The ignored pixel inflates the denominator only. The gradient rewards driving
*every* class probability toward zero there — unsatisfiable under softmax, so
the pressure propagates back into shared decoder features instead.

Two reasons it matters more here than in a toy case. The Cityscapes ignore
region is not scattered noise: it is a coherent block at the bottom of every
frame plus the border, so the same contradictory objective appears in the same
place every iteration — systematic bias, not a wash. And with a double-digit
ignore percentage, `Σ_ignored p²` is a meaningful fraction of the denominator,
deflating every class's Dice and compressing exactly the differences the loss
exists to express.

Measured, not argued (§0.4): perturbing only ignored-region logits moves the
official loss by 0.0146 and leaks gradient with max |g| = 7.5e-6; the masked
version moves by exactly 0.0.

Two further properties to know rather than fix. `_dice_loss` reduces over the
whole batch, so this is batch-level Dice, not a per-image average. And the
denominator uses `Σp²` and `Σg²` rather than `Σp + Σg`; for binary targets
`g² = g`, so only the prediction side differs. The squared form is a legitimate
V-Net variant, not a bug. Keep it for faithfulness or switch to linear for the
standard — just state which.

### The masked implementation

`losses/segmentation_losses.py::MaskedMulticlassDiceLoss`. Decisions, in order:

**Softmax** — applied inside the loss, over the class dimension, on raw logits
(computed in fp32 even under AMP). CE receives the same raw logits separately;
never share a pre-softmaxed tensor.

**One-hot** — `F.one_hot` rather than a 19-iteration Python loop. `F.one_hot`
raises on 255, so labels are first remapped with `torch.where(valid, target, 0)`
and the injected zeros are erased by the mask two lines later.

**Valid-pixel mask** — `m = valid.unsqueeze(1)` multiplies **both** `probs` and
`onehot` before any reduction. This is the entire fix: an ignored pixel
contributes zero to the intersection, to the prediction side of the denominator,
and to the target side, and receives exactly zero gradient (asserted in
`tests/test_losses.py`).

**Intersection / denominator** — reduced over `(0,2,3)` to a per-class vector of
length 19, batch-level, matching EMCAD. Per-image reduction is also defensible
and slightly steadier for rare classes, but diverges from the baseline.

**Absent classes** — in a 512×1024 crop, train, motorcycle and rider are often
absent. For an absent class `intersect = 0` and `onehot.sum = 0`, so
`dice ≈ smooth/(Σp² + smooth)`, near 1 only if the model also predicts almost no
mass there; otherwise the class is penalized against a ground truth that does
not exist in this crop, and the noise scales with how many classes happen to be
missing. `skip_absent_classes=True` averages only over present classes. State
the choice; including all 19 always is what EMCAD does and is equally
defensible given Synapse's more stable class presence.

**Epsilon** — `1e-5`, matching the repo, in both numerator and denominator so an
absent-and-unpredicted class gives `dice ≈ 1` rather than `0/0`. Do not raise it
"for stability": at Cityscapes pixel counts it would be negligible for large
classes and distorting for small ones, exactly backwards for your analysis.

**Companion CE** — `nn.CrossEntropyLoss(ignore_index=255)`, combined as
`0.3·CE + 0.7·Dice`, EMCAD's own weighting (`trainer.py:103`).

**Validation gate** (all implemented in `tests/test_losses.py`, all passing):
ignore-invariance, near-perfect-logits → loss ≈ 0, exactly-zero gradient at
ignored pixels, correct weighting, subset counts 15/4/1, and CE rejecting 255
without `ignore_index`.

### CE vs CE+Dice vs CE+Focal vs CE+Dice+Focal

| | CE | **CE + Dice** | CE + Focal | CE + Dice + Focal |
|---|---|---|---|---|
| Optimizes | mean pixel log-likelihood | likelihood + per-class region overlap | likelihood reweighted to hard pixels | all three |
| Under Cityscapes imbalance | dominated by road/building/sky/veg | Dice is per class then averaged, so rare classes carry weight independent of pixel count | downweights easy (mostly large-class) pixels — indirect | overlapping mechanisms, hard to attribute |
| Alignment with mIoU | weak | **strong** — Dice is a soft relaxation of overlap | moderate | strong but confounded |
| Faithfulness to EMCAD | partial (drops the Dice half of `0.3·CE+0.7·Dice`) | **high — same formulation, same weights** | low | low |
| Implementation risk | lowest | moderate (needs the masked version + its tests) | moderate (needs its own ignore handling, adds γ) | highest |
| Extra hyperparameters | none | none (weights inherited) | γ | γ + three weights |
| Cost per term | cheapest | +1 softmax, +1 one-hot | ≈ CE | highest |

**Baseline: CE.** Simplest correct thing, `ignore_index` built in, and the
reference point every other number is read against. Get the pipeline through the
tiny-overfit gate on CE alone before adding anything — if the overfit fails with
a combined loss you cannot tell whether the bug is in Dice or in the data.

**Primary experiment: CE + Dice.** What the professor asked for, and what keeps
you closest to EMCAD's own loss, so your Cityscapes number is interpretable as
an adaptation of EMCAD rather than a differently-trained model.

**Optional comparison: CE vs CE + Dice** (`configs/cityscapes_pvtv2b2_ce.yaml`
is identical except `loss`). Run it only after the primary is complete.

**Skip focal entirely.** A third variant with a hyperparameter you have no
budget to tune, no advantage in faithfulness, and it would cost time better
spent on the failure analysis. `MaskedFocalLoss` is implemented so you can speak
to it, not so you run it. If asked: *I chose CE+Dice because it matches the
baseline's formulation; focal would have introduced a tuning dimension without
making the comparison to EMCAD cleaner.*
---

## 10. Training-loop design

Implemented in `train_cityscapes.py`. Component by component, with the
reuse/modify/rewrite decision:

| Component | Decision | Detail |
|---|---|---|
| Dataloader | **new** | `datasets/cityscapes.py`; Synapse's is `.npz`/`.h5` volumetric |
| Model creation | **reuse unchanged** | `EMCADNet(num_classes=19, ...)`, same argument names as `train_synapse.py:120` |
| Pretrained loading | **reuse, one-line compat fix** | `networks.py:67-72`; torch ≥ 2.6 needs `weights_only=False` at line 68 |
| Optimizer | **reuse** | AdamW, as `trainer.py:72` |
| LR schedule | **modify** | reinstate the poly decay the repo commented out at `trainer.py:118`, plus linear warmup |
| AMP | **new** | repo imports `torch.cuda.amp` (`trainer.py:17`) but never uses it; this uses `torch.amp` |
| Gradient flow | **reuse** | `zero_grad → backward → step`, with `GradScaler` under AMP |
| Supervision | **reuse logic, reimplemented** | `supervision_subsets` reproduces `trainer.py:91-100` exactly |
| Loss | **modify** | masked Dice + `ignore_index=255` (§9) |
| Logging | **new** | console + `train_log.csv` + `train_summary.json`; no TensorBoard dependency |
| Checkpointing | **modify** | `last.pth` / `best_miou.pth` with full resume state (§17) |
| Validation | **new** | confusion-matrix mIoU, iteration-based, not per-epoch |

Three design choices worth defending:

**Iterations, not epochs.** A bounded session can end mid-epoch. Iteration
counting plus `--max-hours` makes every run resumable at a known point; epochs
make "where was I" ambiguous.

**Poly LR reinstated.** `trainer.py:118-119` has the schedule written and
disabled with the comment "we did not use this." Poly decay is near-universal
for Cityscapes and costs nothing; reinstating it is a deviation to declare, with
that justification.

**Validation on a schedule, not every epoch.** Synapse validates after every
epoch (`trainer.py:136`). On 500 Cityscapes images that is expensive; this runs
every `val_interval` iterations, and `--val-limit` allows a cheap partial
validation during sanity runs.

### Starting values — these are starting points, not facts

Nothing in this table is tuned, published, or measured on your hardware. Each is
a defensible initial choice; record what you actually ran.

| Setting | Start | Reasoning |
|---|---|---|
| Crop | 512×1024 | divisible by 32; half-resolution keeps batch 2 feasible in ~16 GB |
| Batch size | 2 | largest likely to fit at this crop with AMP; raise if `supervision_cost.csv` shows headroom |
| Grad accumulation | 1 | raise to 2–4 to emulate batch 4–8 if OOM forces batch 1 |
| Optimizer | AdamW | EMCAD's own choice (`trainer.py:72`) |
| LR | 6e-5 | EMCAD uses 1e-4 at batch 6; scaled down roughly with batch size. AdamW + ViT encoder is LR-sensitive — if loss plateaus high, this is the first thing to question |
| Weight decay | 0.01 | AdamW default-ish; EMCAD uses 1e-4, which is low for AdamW |
| Warmup | 500 iters linear | transformers destabilize without it |
| Schedule | poly, power 0.9 | Cityscapes convention |
| Max iters | 20,000 | ≈13 epochs at batch 2; an honest baseline, not a leaderboard run |
| AMP | on | roughly halves activation memory; see §15 on comparability |
| Val interval | 2000 | ~10 validation points |
| Seed | 1234 | recorded in config and summary |

Sanity check before launching: measured step time × `max_iters` must fit your
session with slack. If it does not, cut `max_iters` or the crop **before**
starting. `--max-hours` is a safety net, not a plan.

---

## 11. Implementation milestones

The core of the roadmap. Each gate has an objective, the files involved, the
exact command, the expected output, pass criteria, likely failures, and what not
to do until it passes. Later milestones depend only on earlier ones — nothing
here depends on a date.

### M0 — Environment, data, weights

**Objective:** torch + CUDA + timm + `lib/` + Cityscapes + PVTv2-B2 all present
and discoverable.
**Files:** `tools/check_env.py`, `utils/cityscapes_labels.py`.
**Test:** `python -m tools.check_env`
**Expected:** PASS lines for torch, CUDA, timm, `lib/`, EMCADNet build,
pretrained weights, dataset located, free space; `results/environment.json`
written.
**Pass:** every check PASS, and the GPU line shows a real device.
**Likely failures:** accelerator not enabled in the Kaggle pane; `lib/` not
copied; `pvt_v2_b2.pth` not attached; mirror ships `labelIds` only (→
`tools/make_train_ids.py`); torch ≥ 2.6 `weights_only` (→ the notebook's patch
cell).
**Do not** write any training code paths or launch anything until this passes.

### M1 — Model forward at 19 classes

**Objective:** EMCADNet produces four 19-channel full-resolution logit maps at
the real crop size, and a full optimizer step completes.
**Files:** `tests/test_model_smoke.py`, `lib/networks.py`,
`losses/segmentation_losses.py`.
**Test:** `python -m tests.test_model_smoke --crop 512 1024 --batch 2 --amp
--check-divisibility`
**Expected:** four shapes `(2,19,512,1024)`; parameter counts (encoder
24,849,856 / decoder 1,913,515 / heads 19,532); CE sum near 11.78; finite
combined loss; positive finite grad norm; a peak-memory figure; a loud
`RuntimeError` on the non-divisible input.
**Pass:** all asserts pass and the script prints `MODEL SMOKE TEST PASSED`.
**Likely failures:** OOM at batch 2 (drop to 1, or crop 512×512, and record it);
`weights_only` error loading PVTv2; CE far from 11.78 (label range or head init).
**Do not** touch the dataset until the model is known good — debugging both at
once is how days disappear.

### M2 — Dataset verified

**Objective:** labels are correct, pairs align, augmentation is synchronized.
**Files:** `datasets/cityscapes.py`, `tests/test_dataset.py`.
**Test:** `python -m tests.test_dataset --data-root $ROOT | tee
logs/dataset_validation.log`, then **look at**
`figures/sanity_overlays/*.png`.
**Expected:** 10/10 automated checks; overlays written; frequency CSV.
**Pass:** all automated checks, **and** your own eyes confirm mask edges on
object edges in every overlay.
**Likely failures:** counts off (truncated mirror); values 19–33 (reading
labelIds); offset or mirrored overlays (transform desync); values 19–254 after
augmentation (bilinear on the mask).
**Do not** run any training, not even one step, on unverified labels. Everything
downstream inherits this.

### M3 — One real batch end to end

**Objective:** real data, not synthetic tensors, completes
forward → loss → backward → step.
**Files:** `train_cityscapes.py`.
**Test:** `python -W ignore train_cityscapes.py --config
configs/cityscapes_pvtv2b2.yaml --data-root $ROOT --max-iters 5
--val-interval 5 --val-limit 5 --out-dir /kaggle/working/runs/_m3`
**Expected:** the supervision line (`deep_supervision -> 4 loss terms`), the
loss line, a validation mIoU near chance, a checkpoint written.
**Pass:** runs to completion with finite losses and no device-side assert.
**Likely failures:** `IndexError`/device-side assert (a 255 reaching CE without
`ignore_index` — should be impossible here, but check the dataset if it
happens); shape mismatch (crop not divisible by 32); dataloader worker crash
(lower `--num-workers` to 0 to get a readable traceback).
**Do not** interpret the mIoU — five iterations means nothing.

### M4 — Tiny subset overfits

**Objective:** prove the whole chain can memorize. See §12.
**Files:** `tools/tiny_overfit.py`.
**Test:** `python -m tools.tiny_overfit --data-root $ROOT --images 12
--iters 300 --amp`
**Expected:** loss collapsing, per-class IoU table over the subset,
`logs/overfit.json`, `OVERFIT TEST PASSED`.
**Pass:** the three criteria in §12.
**Likely failures:** see §12.
**Do not** start any long run until this passes. This is the single most
important gate in the manual: a model that cannot memorize 12 images will not
learn 2975, and a full run would hide the bug for hours.

### M5 — Metrics validated

**Objective:** the evaluator is provably correct before any number is trusted.
**Files:** `utils/metrics.py`, `tests/test_metrics.py`, `tests/test_losses.py`.
**Test:** `python -m tests.test_metrics && python -m tests.test_losses`
**Expected:** `ALL METRIC TESTS PASSED`, `ALL LOSS TESTS PASSED`, including the
numerical demonstration of the Dice defect.
**Pass:** both suites green; save the output into `logs/`.
**Likely failures:** none expected (both suites pass as shipped) — if one fails
after you edit something, that is the suite doing its job.
**Do not** report any IoU produced by an untested metric.

M5 is independent of M2–M4 (no GPU, no data) and can be run at any point.

### M6 — Profiling works

**Objective:** memory and step time are measured correctly, and the supervision
decision is made from numbers.
**Files:** `utils/profiling.py`, `tools/supervision_cost.py`.
**Test:** `python -m tools.supervision_cost --crop 512 1024 --batch 2 --amp`
**Expected:** `results/supervision_cost.csv` with three rows and a printed
relative comparison; OOM reported rather than crashing.
**Pass:** three rows with finite step times (or an explicit OOM for mutation),
and a decision you can defend.
**Likely failures:** mutation OOM at batch 2 — that is a result, record it.
**Do not** quote any efficiency number obtained without
`torch.cuda.synchronize()` (§15).

### M7 — Short training run behaves

**Objective:** confirm the optimization is actually learning before committing
hours.
**Test:** `python -W ignore train_cityscapes.py --config
configs/cityscapes_pvtv2b2.yaml --data-root $ROOT --max-iters 300
--val-interval 300 --val-limit 50 --out-dir /kaggle/working/runs/sanity`
**Expected:** loss trending down over 300 iterations; a validation mIoU clearly
above chance.
**Pass:** final loss well below the first logged loss, and mIoU above ~0.10
(chance with 19 classes is ~0.05 and a road-only predictor is lower still on
mIoU). Treat this as a trend check, not a quality measurement.
**Likely failures:** flat loss (LR too low or too high — try 3e-5 and 1e-4 and
pick by the curve, which is a sanity check, not a sweep); NaN (see §27).
**Do not** tune hyperparameters here beyond making learning happen.

### M8 — Primary run

**Objective:** the one complete experiment: PVTv2-B2 + EMCAD decoder, 19
classes, CE + masked Dice, Cityscapes train → val.
**Test:** `python -W ignore train_cityscapes.py --config
configs/cityscapes_pvtv2b2.yaml --data-root $ROOT --out-dir
/kaggle/working/runs/primary`, resumed with `--resume .../last.pth` as needed.
**Expected:** periodic validation mIoU rising then flattening;
`train_log.csv`, `train_summary.json`, `best_miou.pth`.
**Pass:** the run reaches `max_iters` (across however many sessions) with a
monotone-ish best mIoU and no crash.
**Likely failures:** session killed (resume); disk full (keep only `last` and
`best`); mIoU plateauing very low (stop and revisit M4 rather than training
longer).
**Do not** change anything mid-run. A configuration changed halfway makes the
result unreportable. If you must change something, start a new run directory.

### M9 — Final validation

**Objective:** the reported numbers, produced by one command from one
checkpoint.
**Test:** `python -W ignore eval_cityscapes.py --checkpoint
/kaggle/working/runs/primary/best_miou.pth --data-root $ROOT --out-dir results
--save-preds results/preds_trainids`
**Expected:** per-class table printed; `results/per_class_iou.csv`,
`summary.json`, `confusion_matrix.npy`.
**Pass:** `n_valid_classes == 19` (all 19 appear in the official val split — if
fewer, investigate before reporting), and mIoU consistent with the last
training-time validation.
**Likely failures:** a large gap between training-time and final mIoU usually
means a resize difference between the two paths; check `val_resize` in both.
**Do not** re-run with different settings and report the better number. Decide
the protocol first (native or resized), then run once.

### M10 — Qualitative analysis

**Objective:** figures and a quantitative failure analysis that agree with the
metrics.
**Test:** `python -W ignore visualize_cityscapes.py --checkpoint ...
--num-images 8 --strategy spread` then `python -m tools.analyze_failures`
**Expected:** 5-panel figures, `qualitative_manifest.json`,
`failure_analysis.json`, `thin_vs_large.csv`, two analysis figures.
**Pass:** panels use the same checkpoint and the same resize as M9; the manifest
records the selection strategy.
**Likely failures:** figures produced at a different resolution than the metrics
(then they cannot be cited as evidence for them).
**Do not** select only the best-looking scenes; `--strategy spread` samples
across the image-mIoU range for exactly this reason, and the manifest proves it.

### M11 — Repository and report

**Objective:** a stranger can reproduce the run from the repository alone.
**Test:** the §28 master checklist, plus a clean-clone read-through.
**Pass:** README tables filled from `results/`, logs committed, exact commands
present, `RESEARCH_SUMMARY.md` written, every deviation declared.
**Do not** leave placeholder numbers anywhere. An unfilled template is worse
than an omission.

---

## 12. The tiny-overfit experiment

**Images:** 12 (range 8–20). Enough that it is not a single-image fluke, few
enough to memorize in a few hundred iterations.

**Sampling:** a fixed seed shuffles the full train list and takes the first 12,
so the subset is reproducible and the exact names are printed and stored in
`logs/overfit.json`. Not "the first 12 alphabetically," which would be one city
and one lighting condition.

**Augmentation: OFF.** `augment=False` gives a deterministic centre crop.
Augmentation is a regularizer; leaving it on fights memorization and muddies the
signal. This is a correctness probe, not a generalization measurement.

**Evaluation: on the same 12 images.** Deliberate, and stated in the README so
nobody mistakes it for a result.

**Why a flat "mIoU > 0.90" is the wrong criterion.** A 12-image subset will not
contain all 19 classes, and some present classes will have a handful of pixels.
Averaging over 19 would cap the achievable value below the threshold by
construction, and averaging over every class with a single pixel present makes
the metric noise-dominated.

**The criterion actually used** (`tools/tiny_overfit.py`), all three required:

1. **Loss collapse** — final loss < 25% of the first logged loss.
2. **Subset mIoU > 0.85**, averaged **only over classes with ≥ 2000 pixels in
   the subset** (`--min-pixels`). The script prints which classes were scored,
   which were present but too rare, and which were absent, so the denominator is
   explicit and reportable.
3. **Coverage** — at least 5 classes scored, and at least half of the present
   classes, so the number is not resting on road and sky alone.

**Interpreting failure:**

| Symptom | Most likely cause |
|---|---|
| Loss decreases but subset mIoU stays low | prediction/label misalignment — go back to M2 overlays |
| Loss flat from the first iteration | LR far too low, or gradients not reaching the encoder; check `requires_grad` and the grad norm from M1 |
| Loss collapses but only large classes are scored | fine, if coverage passes; if not, the subset is unrepresentative — re-sample with a different seed |
| Loss NaN | AMP + an unstable loss term; re-run with `--no-amp` to localize |
| Loss collapses with `--loss ce` but not `ce_dice` | the Dice term — re-run `tests/test_losses.py` |
| Needs far more than 300 iterations | pretrained weights probably did not load; check the M0 line |

Failing to overfit means something in the chain is wrong. It does not mean "the
model needs more training." Fix the chain.

---

## 13. IoU / mIoU implementation

`utils/metrics.py`. The pipeline:

**Prediction.** `logits = P[-1]` (the finest head, matching
`utils/utils.py:268`), then `argmax` over the class dimension in fp32, giving
`[B,H,W]` of trainIds.

**Resizing.** Prediction and ground truth must be at the same resolution. Two
valid protocols: evaluate at native 1024×2048 (standard — the preferred option),
or evaluate at the reduced size with the **mask downsampled by nearest** (what
`--val-resize` does). Never upsample a prediction bilinearly and argmax
afterwards without saying so, and never compare a resized prediction against a
native mask.

**Ignore mask.** Pixels with `target == 255` are dropped before accumulation —
not zeroed, dropped. Both prediction and target are then range-asserted, so an
out-of-range prediction raises instead of silently wrapping in `bincount`.

**Confusion matrix.** `M += np.bincount(19*gt + pred, minlength=361).reshape(19,19)`
over valid pixels, accumulated across the whole split as int64 (no overflow:
500 × 2M pixels ≈ 1e9, far inside int64).

**Derivations.** With rows = GT, columns = prediction:

```
TP_c  = M[c, c]
FN_c  = M[c, :].sum() - M[c, c]
FP_c  = M[:, c].sum() - M[c, c]
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU  = mean over classes with (TP+FP+FN) > 0
```

`precision_c = TP/(TP+FP)` and `recall_c = TP/(TP+FN)` come free and are worth
reporting for thin classes in §20 — they separate "predicts poles everywhere"
from "misses poles."

**Absent from a batch vs absent from the split.** A class absent from one batch
needs no special handling at all, because the matrix accumulates and presence is
only decided at the end. A class absent from the entire split has
`TP+FP+FN == 0`; its IoU is NaN and it is excluded from the mean, matching the
official evaluator. All 19 classes do appear in the official val split, so
`n_valid_classes` should read 19 — anything less is a signal, which is why the
summary records it.

**Unit tests** (`tests/test_metrics.py`, all passing as shipped):

| Test | Asserts |
|---|---|
| perfect prediction | every present IoU = 1.0, mIoU = 1.0, pixel acc = 1.0 |
| all predictions wrong (`(gt+1) % 19`) | every IoU = 0.0, mIoU = 0.0 |
| one-class swap (5 ↔ 7) | exactly those two drop to 0, all others stay 1.0 |
| all-ignore target | empty matrix, NaN mIoU, no division by zero |
| mixed ignored/valid | matrix identical whatever is predicted in the ignored region |
| absent classes | NaN, excluded from the mean, counted in `n_valid_classes` |
| hand-computed 4-pixel case | IoU = 1/3 exactly |
| incremental accumulation | batched updates equal a single-shot update |

The last two are not in your list but earn their place: the hand-computed case
catches a transposed matrix (which the symmetric synthetic tests would not), and
the accumulation test catches a reset bug across batches.

---

## 14. Cross-checking against the official Cityscapes evaluation

Independent validation of your evaluator, and the cheapest credibility you can
buy.

**Prediction format.** The official evaluator
(`cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling`) reads PNG files
whose pixel values are **labelIds (0–33), not trainIds**. It matches them to
ground truth by filename, at native 1024×2048.

**ID space.** You must convert: trainId → labelId (0→7 road, 1→8 sidewalk, 2→11
building, …, 18→33 bicycle), with any ignored prediction mapped to 0
(`unlabeled`). `tools/official_eval_crosscheck.py` builds that inverse LUT from
the same table and does the conversion.

**Resolution.** The official tool compares at native resolution. For the
cross-check to be meaningful, produce predictions with `eval_cityscapes.py`
**without** `--val-resize`, or upsample them to native with **nearest** first.
This is the most common source of a spurious mismatch.

**What to compare.** The official output's "average" IoU-class figure against
your `summary.json → miou`, and the official per-class column against
`per_class_iou.csv`.

**Procedure:**

```bash
python eval_cityscapes.py --checkpoint runs/primary/best_miou.pth \
    --data-root $ROOT --out-dir results/native --save-preds results/preds_trainids
pip install cityscapesscripts          # needs internet
python -m tools.official_eval_crosscheck --pred-dir results/preds_trainids \
    --data-root $ROOT --custom-summary results/native/summary.json --run-official
```

**Interpretation.** Agreement to ~1e-4 means your confusion matrix is correct —
say so explicitly in the README, it is a strong statement. A systematic gap
points, in order of likelihood, to: a resize mismatch, an ignore-handling
difference, or a class-ordering error. A gap on exactly one or two classes
points at the trainId→labelId mapping for those classes.

If internet is unavailable in your session, the conversion step still runs
offline and you can note that the cross-check is prepared but not executed. That
is honest and still better than nothing.

---

## 15. Training-efficiency profiling

### A. Peak GPU memory

```python
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()      # immediately before the measured window
...                                        # the training steps
torch.cuda.synchronize()
alloc = torch.cuda.max_memory_allocated() / 1024**3
resv  = torch.cuda.max_memory_reserved()  / 1024**3
```

Report **`max_memory_allocated`** as the headline (tensor memory PyTorch
actually held) and `max_memory_reserved` alongside (what the caching allocator
took from the driver). They differ substantially and quoting the wrong one
invites a correction. `nvidia-smi` is the wrong instrument entirely: it shows
reserved memory plus the CUDA context plus anything else on the device.

Reset the peak **after** warm-up if you want steady-state memory, or before the
first iteration if you want the true peak including allocator growth. This
project resets once before training and reports the overall peak, which is the
more conservative claim. State which you did.

### B. Step time

**Definition used here, which you should state verbatim in the README:**

> Step time is the wall-clock duration of forward + loss + backward +
> optimizer.step + zero_grad, measured with `torch.cuda.synchronize()`
> immediately before starting and immediately after finishing the timed GPU
> work, averaged over iterations after a 20-iteration warm-up, excluding
> dataloading and excluding validation.

| Included | Excluded |
|---|---|
| forward | dataloading / host-to-device transfer |
| loss (all supervision terms) | validation |
| backward | checkpoint writing |
| optimizer step + zero_grad | logging |

Dataloading is excluded because it is CPU/IO-bound and depends on worker count
and disk speed, which would make the number describe the machine more than the
model. Excluding it silently would be hiding something, so `data_wait_s` is
logged next to every step time — the exclusion is visible, and if data wait
dominates you can see that too.

**Warm-up.** The first iterations include CUDA context creation, cuDNN
algorithm selection, allocator growth and lazy module init, and can be several
times slower. `StepTimer(warmup=20)` discards them and reports
`warmup_iterations` so the exclusion is auditable.

**Averaging.** Report mean and standard deviation over the timed iterations plus
the count. A mean without a count is not a measurement.

### AMP and comparability

AMP changes both numbers, so it must be reported with them, never alone. Two
specifics worth knowing:

- Memory falls roughly with activation precision, but not by exactly half:
  parameters, optimizer states (AdamW keeps two fp32 moments) and fp32 master
  weights are unaffected.
- Speed depends on the GPU. T4 has tensor cores and benefits substantially;
  P100 has no tensor cores (its fp16 advantage comes from packed fp16
  arithmetic) and gains much less. So an AMP step time from a T4 is not
  comparable to one from a P100, and neither is comparable to a non-AMP number.

**Rule: never compare efficiency numbers across different GPU/AMP/crop/batch
configurations.** Everything in `summary.json` that pins those down —
`gpu_name`, `torch_version`, `cuda_version`, `driver`, `crop_size`,
`batch_size`, `amp` — exists so a reader can tell whether two numbers are
comparable. For the supervision comparison specifically, `supervision_cost.csv`
holds all three modes measured in one session on one GPU with one setting,
which is the only way that comparison means anything.

---

## 16. Training logging

Three sinks, no external services:

**Console** — one line every `log_interval` iterations:

```
iter   1500/20000 loss 0.5312 ce 0.4120 dice 0.5823 lr 5.73e-05 step 0.412s data 0.031s mem 9.84GB
[val] iter 2000 mIoU 0.3914 (best 0.3914) peak_alloc 9.84GB
```

**CSV** — `runs/<exp>/train_log.csv`, one row per logged iteration and one per
validation:

| column | meaning |
|---|---|
| `iter` | global step |
| `epoch_frac` | iterations × batch / dataset size, so epochs are still readable |
| `total_loss`, `ce`, `dice`, `focal` | components, averaged over supervision terms |
| `lr` | current poly/warmup value |
| `step_time_s` | as defined in §15 |
| `data_wait_s` | the excluded dataloading time, visible |
| `peak_mem_gb` | running peak allocated |
| `val_miou`, `best_miou` | filled on validation rows |

CSV because it opens in anything, diffs in git, and plots in three lines of
pandas.

**JSON** — `runs/<exp>/train_summary.json` at the end: iterations completed,
best mIoU, peak allocated and reserved memory, mean/std/count step time, warm-up
count, AMP, batch, crop, supervision mode, number of loss terms, wall-clock
hours, the full config, and the environment block. This is the file the README
efficiency table is filled from.

**Train mIoU:** not logged per iteration. Computing it properly means a
confusion matrix over training crops, which costs time and measures memorization
more than anything else. If you want it, run `validate()` against a fixed
subset of the *train* split at the same interval, and label it clearly.

**TensorBoard:** the repo uses `tensorboardX` (`trainer.py:13,73`). Deliberately
dropped here — it adds a dependency, and the CSV plus matplotlib covers
everything. If you want curves in the README, plot from the CSV and commit the
PNG.

---

## 17. Checkpoint strategy

**Files kept** (`/kaggle/working` is space-limited, so only two by default):

| File | Written when | Purpose |
|---|---|---|
| `last.pth` | every validation | resume after a session ends |
| `best_miou.pth` | validation mIoU improves | the checkpoint you evaluate and report |
| `iter_<n>.pth` | optional, manual | only if you need a mid-training comparison |

A full checkpoint of this model is roughly 100 MB of weights plus AdamW's two
moment tensors, so budget around 300–350 MB each. Two is affordable; ten is not.

**Metadata stored with every checkpoint** (`train_cityscapes.py`):

```python
{"model": state_dict, "optimizer": optimizer.state_dict(),
 "scaler": scaler.state_dict(), "iteration": it,
 "best_miou": ..., "val_miou": ..., "config": cfg, "env": env_info()}
```

Scheduler state is not stored because the poly LR is computed from `iteration`
alone — a stateless schedule is one less thing to desynchronize on resume, and
`iteration` already carries it.

Carrying `config` and `env` inside the checkpoint means `eval_cityscapes.py` can
rebuild the exact model without being told the architecture flags again, and a
checkpoint found later is self-describing.

**Resuming reproducibly:**

```bash
python train_cityscapes.py --config configs/cityscapes_pvtv2b2.yaml \
    --data-root $ROOT --out-dir runs/primary --resume runs/primary/last.pth
```

restores weights, optimizer moments, AMP scaler, iteration counter and best
mIoU, and recomputes the LR from the restored iteration.

**What resume does not restore, and the honest caveat:** the dataloader's
shuffling position and the Python/NumPy/CUDA RNG states. After a resume the
sample order differs from an uninterrupted run, so a resumed run is not
bit-identical to a single-session run of the same length. Say so in the
README's reproducibility section rather than claiming exactness you do not have.
Storing RNG states is possible but adds failure modes under a different worker
count, which a resumed Kaggle session may well have.

---

## 18. Qualitative visualization pipeline

`visualize_cityscapes.py` produces one row per scene:

**Input RGB | Ground truth | Prediction | Prediction overlay | Error map**

- **RGB** — denormalized from the exact tensor the model saw, so what you look
  at is what it was given.
- **GT / Prediction** — official Cityscapes palette, ignore rendered black.
- **Overlay** — 55% RGB + 45% prediction colour, which shows boundary quality
  against real structure better than either panel alone.
- **Error map** — the panel that carries the analysis:

| Colour | Meaning |
|---|---|
| grey | ignored (GT = 255), excluded from metrics |
| white | correct |
| **red** | wrong, predicted class is in the focus set → **thin-class false positive** |
| **blue** | wrong, GT class is in the focus set → **thin-class false negative (a miss)** |
| orange | wrong, neither side in the focus set → generic confusion |

The red/blue split is the point. "Poles are bad" is not an observation; "poles
are predominantly blue, so the model misses them rather than hallucinating
them" is, and it pairs directly with the precision/recall columns in
`thin_vs_large.csv`. The focus set defaults to the seven thin/small classes the
assignment names.

**Composite panels, not individual images.** A reader compares across columns at
a glance, and one file per scene keeps `figures/` navigable. Individual images
would be right only if you needed to zoom on one region, which is better served
by a crop of the panel.

**Guaranteeing figures and metrics agree** — three mechanisms:

1. `visualize_cityscapes.py` imports `load_model` from `eval_cityscapes.py`, so
   the model construction and checkpoint loading are literally the same code.
2. Prediction uses the same `P[-1]` argmax and the same `--val-resize`.
3. `figures/qualitative_manifest.json` records the checkpoint path, the resize,
   the selection strategy and each scene's own mIoU, so a reader can verify the
   figures came from the reported run.

Run visualization **after** M9 with the same checkpoint and the same resize. If
you evaluate at native resolution and visualize at half, say so — the panels are
then illustrative rather than evidential.

**Scene selection.** `--strategy spread` ranks the whole val split by image mIoU
and samples evenly across that range, which is the defensible default: it
guarantees the figures are not all best cases, and the manifest proves it.
`--strategy worst` and `most_thin_classes` are useful supplements for the
failure section. Using only `best` would be misrepresentation.

---

## 19. Failure-mode analysis framework

A framework, not a list of impressions: every observation is a record with
evidence attached, and the quantitative and qualitative sides must agree.

### The record

For each selected scene, inspect the eleven conditions below and record:

| Field | Content |
|---|---|
| `scene` | image name, from the manifest |
| `class` | Cityscapes class |
| `visual_condition` | what is true of the scene — distance, scale, occlusion, lighting, clutter |
| `prediction_behavior` | what the model did: fragmented, dissolved into background, confused with a specific class, boundary displaced |
| `candidate_explanation` | hedged, one sentence |
| `evidence` | the panel, plus the per-class IoU / precision / recall / top-confusion entry that supports it |

### The eleven conditions

| # | Condition | What to look for | Quantitative partner |
|---|---|---|---|
| 1 | road/sidewalk boundary | displaced or wandering boundary on a flat colour transition | IoU 0 and 1; `top_confusions[road]` |
| 2 | poles | fragmentation into dashes, or disappearance at distance | IoU 5; recall vs precision |
| 3 | traffic signs | detected but wrong extent, or confused with pole | IoU 7; confusion 7↔5 |
| 4 | traffic lights | small distant instances missed entirely | IoU 6; recall 6 |
| 5 | pedestrians | merged groups, thin limbs lost | IoU 11; confusion 11↔12 |
| 6 | riders | confused with person, both directions | IoU 12; confusion 12↔11 |
| 7 | motorcycles | fragmentation, confusion with bicycle | IoU 17; confusion 17↔18 |
| 8 | bicycles | spokes/frame lost; confusion with motorcycle or rider | IoU 18 |
| 9 | distant vehicles | small cars merged or dropped | IoU 13 vs 14/15/16 |
| 10 | occlusion | partially hidden objects split into pieces | per-image mIoU on those scenes |
| 11 | thin structures generally | fences and poles against cluttered backgrounds | group mIoU, `small_thin` vs `large_region` |

### How the two sides support each other

`tools/analyze_failures.py` produces the quantitative half from
`confusion_matrix.npy`: per-class IoU, precision, recall, the top-3 confusions
per class with the share of that class's GT pixels going each way, group mIoU,
and the correlation between log pixel frequency and IoU.

The rule for writing it up: **every visual claim must be paired with a number,
and every number worth mentioning should have a panel.** "The model confuses
rider and person" becomes "rider IoU is X; the confusion table shows Y% of rider
ground-truth pixels predicted as person, and the panel for `<scene>` shows this
on a cyclist at mid-distance." One is an impression; the other is a finding.

### Language

The professor was explicit. Use *may indicate*, *suggests*, *is consistent
with*. Reserve *shows* and *demonstrates* for things your experiments actually
established.

Established by your experiments: the per-class IoU values; the confusion
patterns; the group gap; the frequency/IoU correlation; the memory and timing
numbers; that the architecture needed no rewrite.

**Not** established, and must be hedged: that stride-4 features are *the reason*
poles fragment; that the 7×7 SAB kernel is *why* thin structures suffer; that
Dice *caused* any particular class improvement without the CE-only ablation to
compare against; that mutation supervision *would* improve the result.

A correlation between frequency and IoU is descriptive. Frequency and object
size are themselves correlated in Cityscapes, so the data cannot separate them —
say that, and you have demonstrated more judgement than a confident causal claim
would.
---

## 20. Large classes vs small/thin classes

The professor asked for this comparison explicitly. Two halves, kept separate
from the official metric.

### Grouping

Three groups, defined in `utils/cityscapes_labels.py::CLASS_GROUPS`:

| Group | trainIds | Classes |
|---|---|---|
| `large_region` | 0,1,2,8,9,10 | road, sidewalk, building, vegetation, terrain, sky |
| `medium_object` | 3,4,13,14,15,16 | wall, fence, car, truck, bus, train |
| `small_thin` | 5,6,7,11,12,17,18 | pole, traffic light, traffic sign, person, rider, motorcycle, bicycle |

Three groups rather than two because a binary split would force car (large,
compact, common) and pole (tiny, thin, common) into the same bucket as bus
(large but rare), which hides the distinction instead of showing it. The middle
group also gives the comparison a monotone trend to point at rather than a
single gap.

**This grouping never touches official mIoU.** `summary.json → miou` is always
the unweighted mean over all 19 classes. The group means live in a separate
field (`group_miou`) and a separate CSV. Say this explicitly in the README —
inventing a favourable aggregation is the kind of thing that ends an interview
badly, and pre-empting the suspicion is free.

### Quantitative comparison

`results/thin_vs_large.csv`, one row per class: group, trainId, name, IoU,
precision, recall, train pixel fraction.

Report:

1. **Mean IoU per group**, with the class list so the reader can recompute it.
2. **Per-class IoU sorted within each group**, so the ordering inside a group is
   visible rather than averaged away.
3. **Precision and recall for the thin classes.** This is the most informative
   column pair in the whole analysis: low recall with decent precision means the
   model misses thin objects; low precision with decent recall means it
   hallucinates them. These demand different explanations and the mIoU number
   cannot distinguish them.
4. **IoU vs pixel frequency** (`figures/iou_vs_frequency.png`, log-x, coloured
   by group, with the Pearson r on log-frequency) — one figure that makes the
   whole argument visible.
5. **Top confusions per thin class** from `failure_analysis.json`: where do
   pole pixels actually go?

### Qualitative comparison

Two or three panels chosen to show the contrast directly: a scene where road,
building and sky are near-perfect while poles and signs in the same frame are
fragmented. The error map makes this immediate — large white regions with red
and blue speckle concentrated on thin structures.

Then state the honest limitation: IoU penalizes a fixed boundary error far more
for a thin object than a large one, purely by area arithmetic. A 2-pixel
boundary error on a 6-pixel-wide pole can halve its IoU; the same error on a
building is invisible. So part of the measured gap is metric geometry rather
than model failure. Saying that unprompted is exactly the kind of reasoning the
task is probing for, and it is why the optional boundary-IoU/trimap analysis
(O3) exists — it is the measurement that would separate the two, and you can
name it as future work without running it.

---

## 21. Experiment design

The minimum scientifically useful set, and nothing more.

### E0 — Primary: CE + masked Dice (required)

- **Question:** can EMCAD, adapted correctly, perform 19-class Cityscapes
  segmentation, and what are its per-class IoU, memory and step time?
- **Changed:** nothing; this is the reference.
- **Controlled:** PVTv2-B2 pretrained, EMCAD decoder unmodified, 512×1024,
  batch 2, AdamW 6e-5 poly, deep supervision, seed 1234, AMP on.
- **Gained:** every required deliverable. Without it nothing else exists.

### E1 — Supervision-mode cost (strongly recommended, minutes not hours)

- **Question:** what do EMCAD's three supervision modes actually cost at
  Cityscapes resolution?
- **Changed:** supervision mode only.
- **Controlled:** same model, crop, batch, AMP, same session, same GPU.
- **Gained:** turns the single most questionable decision in the project into a
  measurement. Costs about ten training steps per mode. `tools/supervision_cost.py`.

### E2 — One additional *training* run, if compute allows exactly one

Two candidates:

| | CE vs CE+Dice | deep_supervision vs mutation |
|---|---|---|
| Question | does the Dice term help, and which classes? | does EMCAD's own supervision scheme matter at this resolution? |
| Faithfulness relevance | moderate — both are EMCAD-compatible | **high — this is the deviation you will be asked about** |
| Cost | one run of equal length | one run, substantially slower per step, possibly OOM |
| Risk | low | moderate — may not fit, and may need a reduced crop, breaking the control |
| Interview value | good | **better, if it completes** |

**Recommendation: if exactly one additional run is possible, run
CE vs CE+Dice** (`configs/cityscapes_pvtv2b2_ce.yaml`, identical except the
loss). Reasons: it is the comparison the professor explicitly named as the
optional one; it is guaranteed to fit, because it is strictly cheaper than the
primary; it isolates a single variable cleanly; and it speaks to the
thin-class story, since Dice is the component meant to help rare classes — you
can report **per-class** deltas, not just a mIoU delta, which is a much richer
finding.

The supervision comparison is more informative in principle, but it is the one
most likely to OOM or to require a crop change that destroys the control. E1
already gives you the cost side of that question for free, which is the part
you actually need to defend the decision. If E1 shows mutation fits comfortably,
then reconsider: in that case train the primary with mutation and the question
disappears entirely.

### What not to run

Encoder swaps (B0/B1/B5, ResNet), kernel-size variants, `--concatenation`,
LGAG kernel sizes, learning-rate sweeps, longer schedules chasing mIoU,
multi-scale or sliding-window inference. Each costs a run and answers a
question nobody asked. The assignment rewards one correct experiment with a
careful analysis over four shallow ones.

---

## 22. Repository structure

```
EMCAD-Cityscapes/
  configs/
    cityscapes_pvtv2b2.yaml          primary experiment
    cityscapes_pvtv2b2_ce.yaml       CE-only ablation (identical but `loss`)
  datasets/
    cityscapes.py                    Dataset, joint transforms, root discovery
  losses/
    segmentation_losses.py           masked Dice, focal, combined, supervision subsets
  lib/                               UNMODIFIED from SLDGroup/EMCAD
    networks.py decoders.py pvtv2.py resnet.py
  utils/
    cityscapes_labels.py             34-row official table, palette, groups
    metrics.py                       confusion matrix, IoU, mIoU, CSV/JSON export
    profiling.py                     env info, peak memory, StepTimer
  tests/
    test_model_smoke.py              M1 gate
    test_dataset.py                  M2 gate (8 checks + synthetic mode)
    test_metrics.py                  M5 gate
    test_losses.py                   M5 gate + the Dice-defect demonstration
  tools/
    check_env.py                     M0 gate
    make_train_ids.py                labelIds -> labelTrainIds, offline-safe
    tiny_overfit.py                  M4 gate
    supervision_cost.py              E1
    analyze_failures.py              quantitative failure analysis
    official_eval_crosscheck.py      §14
  kaggle/
    EMCAD_Cityscapes_Kaggle.ipynb    gate-by-gate runner
    build_notebook.py                generator (keeps the notebook diffable)
  scripts/
    train.sh  evaluate.sh
  logs/          dataset_validation.log, overfit.json, train_log.csv, test output
  results/       per_class_iou.csv, summary.json, confusion_matrix.npy,
                 class_frequency.csv, failure_analysis.json, thin_vs_large.csv,
                 supervision_cost.csv, environment.json
  figures/       qualitative_*.png, sanity_overlays/, iou_vs_frequency.png,
                 confusion_heatmap.png, qualitative_manifest.json
  train_cityscapes.py  eval_cityscapes.py  visualize_cityscapes.py
  requirements.txt  README.md  MANUAL.md  RESEARCH_SUMMARY.md  LICENSE  NOTICE
```

### Inherited / modified / new

| Status | Files |
|---|---|
| **Inherited unchanged** | `lib/networks.py`, `lib/decoders.py`, `lib/pvtv2.py`, `lib/resnet.py` |
| **Inherited with one compatibility fix** | `lib/networks.py:68` only if torch ≥ 2.6 requires `weights_only=False` — a loading fix, not a model change, and it must be declared |
| **Logic reused, reimplemented in a new file** | training loop (from `trainer.py`), supervision subsets (from `trainer.py:91-100` + `powerset`), Dice (from `utils/utils.py:102-138`, corrected), `P[-1]` inference convention (from `utils/utils.py:268`), dataset contract (from `dataset_synapse.py`) |
| **New** | everything in `datasets/`, `losses/`, `utils/`, `tests/`, `tools/`, `configs/`, `kaggle/`, the three top-level scripts |

Keeping `lib/` byte-identical is a deliberate, checkable claim. Verify it before
submitting:

```bash
git clone --depth 1 https://github.com/SLDGroup/EMCAD.git /tmp/EMCAD_ref
diff -r lib /tmp/EMCAD_ref/lib    # expect no output, or only the weights_only line
```

Put that command in the README. It converts "I didn't change the architecture"
from an assertion into something a reader can confirm in ten seconds.

### Fork or separate repository?

**Create a new repository**, not a fork. Reasons: a fork's default comparison
view is a diff against upstream, which buries your work among EMCAD's files; the
deliverable is a Cityscapes study, not a patch to EMCAD; and a fork invites the
reading that you modified the architecture when the whole point is that you did
not. A clean repository with `lib/` vendored and clearly labelled makes the
boundary obvious.

### Crediting the original

Four places, all cheap:

1. **README header** — "Built on the official EMCAD implementation
   (SLDGroup/EMCAD), Rahman, Munir & Marculescu, CVPR 2024", with links to the
   repo and the paper, stated before anything else.
2. **`NOTICE`** — which files are vendored from EMCAD, that they are
   unmodified (or the exact one-line exception), and the upstream commit hash
   you cloned. Record the hash: `git -C /tmp/EMCAD rev-parse HEAD`.
3. **`LICENSE`** — keep EMCAD's license file for the vendored `lib/` and state
   the license of your own additions.
4. **File headers** — every file that reimplements EMCAD logic names the source
   file and lines it derives from, as the files in this repo already do.

Pay particular attention here: the author of the code you are building on is the
person evaluating you.

---

## 23. Reproducibility checklist

| Item | Where | Materially affects reproducibility? |
|---|---|---|
| Random seed | `configs/*.yaml → seed`, echoed into `config.json` and `summary.json` | **Yes** — different seeds move mIoU by a non-trivial amount on a short schedule |
| `cudnn.deterministic` / `benchmark` | `train_cityscapes.py::set_seed` | **Yes** — non-deterministic kernels change results run to run |
| Dataloader `worker_init_fn` + generator | `train_cityscapes.py` | **Yes** — otherwise worker RNGs differ per run |
| `num_workers` | config | **Yes, indirectly** — changes the order in which worker RNG streams are consumed |
| torch / CUDA / cuDNN / driver versions | `results/environment.json`, inside every checkpoint | **Yes** — kernel selection and numerics differ across versions |
| GPU model | same | **Yes** for timing and memory; mildly for numerics |
| AMP on/off | config + summary | **Yes** — fp16 accumulation differs |
| Pretrained checkpoint identity | §below | **Yes** — a different PVTv2-B2 copy is a different experiment |
| Dataset version and mask source | `train_set.mask_kind` printed; record it in the README | **Yes** — official vs self-generated trainIds |
| Crop size, batch size, grad accumulation | config | **Yes** |
| Exact CLI commands | README + `scripts/*.sh` | Yes, for a third party |
| Resume boundaries | `train_log.csv` shows where a run restarted | **Yes** — see the §17 caveat |
| Library versions | `requirements.txt`, pinned after the fact | Yes |

**Checksum the pretrained weights.** A single line removes a whole class of
ambiguity:

```bash
sha256sum pretrained_pth/pvt/pvt_v2_b2.pth | tee results/pretrained_sha256.txt
```

Record it in the README with the source you obtained it from (the PVT GitHub
release or the Google Drive link in EMCAD's README). "PVTv2-B2 pretrained" alone
is not reproducible; a hash is.

**Also record the upstream EMCAD commit:**

```bash
git -C /tmp/EMCAD rev-parse HEAD | tee results/emcad_commit.txt
```

**The honest limitations paragraph**, which belongs in the README:

> Runs are seeded and `cudnn.deterministic` is enabled, so a single-session run
> should reproduce closely on the same GPU with the same library versions. Two
> caveats: resuming does not restore dataloader shuffling or RNG state, so a
> run completed across several sessions will not be bit-identical to an
> uninterrupted one; and results are not expected to be bit-identical across
> different GPU models or PyTorch versions.

Claiming less than you can prove is the correct move. An overclaimed
reproducibility statement is easy to falsify and costs more credibility than
the caveat does.

---

## 24. Final result tables

Copy-paste templates. `<...>` is filled from the named file. **Never leave a
placeholder in a submitted README.**

### A. Experimental configuration

```markdown
| Item | Value |
|---|---|
| Architecture | EMCAD decoder (unmodified) + PVTv2-B2 encoder |
| Source | SLDGroup/EMCAD @ `<results/emcad_commit.txt>` |
| Classes | 19 (Cityscapes trainIds), ignore index 255 |
| Pretrained encoder | PVTv2-B2, sha256 `<results/pretrained_sha256.txt>` |
| Label source | `<official labelTrainIds | generated with tools/make_train_ids.py>` |
| Train / val images | 2975 / 500 |
| Crop (train) | 512×1024, random scale 0.5–2.0, random crop, h-flip, colour jitter |
| Resolution (val) | `<native 1024×2048 | 512×1024 resized>` |
| Batch size | `<cfg>` (grad accumulation `<cfg>`) |
| Optimizer | AdamW, lr `<cfg>`, weight decay `<cfg>` |
| Schedule | `<n>` iterations, poly power 0.9, `<n>` warmup |
| Loss | `0.3·CE(ignore_index=255) + 0.7·masked multiclass Dice` |
| Supervision | `<mode>` (`<n>` loss terms) |
| AMP | `<on/off>` |
| Seed | `<cfg>`, `cudnn.deterministic=True` |
| Hardware | `<summary.json: gpu_name>`, torch `<...>`, CUDA `<...>`, driver `<...>` |
```

### B. Overall results

```markdown
| Metric | Value |
|---|---|
| mIoU (19 classes) | `<summary.json: miou>` |
| Pixel accuracy | `<summary.json: pixel_accuracy>` |
| Classes counted | `<n_valid_classes>` / 19 |
| Evaluated head | `P[-1]` (finest), matching official EMCAD inference |
| Checkpoint | `best_miou.pth` @ iteration `<...>` |
```

### C. Per-class IoU

```markdown
| trainId | Class | IoU (%) | Precision | Recall | Train pixel share (%) |
|---|---|---|---|---|---|
| 0 | road | | | | |
| 1 | sidewalk | | | | |
| 2 | building | | | | |
| 3 | wall | | | | |
| 4 | fence | | | | |
| 5 | pole | | | | |
| 6 | traffic light | | | | |
| 7 | traffic sign | | | | |
| 8 | vegetation | | | | |
| 9 | terrain | | | | |
| 10 | sky | | | | |
| 11 | person | | | | |
| 12 | rider | | | | |
| 13 | car | | | | |
| 14 | truck | | | | |
| 15 | bus | | | | |
| 16 | train | | | | |
| 17 | motorcycle | | | | |
| 18 | bicycle | | | | |
| | **mIoU** | | | | |
```

Fill from `results/per_class_iou.csv` + `results/thin_vs_large.csv`.

Supplementary grouping (state that it does not affect mIoU):

```markdown
| Group | Classes | Mean IoU (%) | Mean train pixel share (%) |
|---|---|---|---|
| Large regions | road, sidewalk, building, vegetation, terrain, sky | | |
| Medium objects | wall, fence, car, truck, bus, train | | |
| Small / thin | pole, traffic light, traffic sign, person, rider, motorcycle, bicycle | | |
```

### D. Efficiency

```markdown
| Measurement | Value |
|---|---|
| Peak GPU memory (allocated) | `<...>` GiB |
| Peak GPU memory (reserved) | `<...>` GiB |
| Mean training step time | `<...>` s ± `<...>` over `<n>` iterations |
| Warm-up iterations excluded | `<...>` |
| Step time definition | forward + loss + backward + optimizer step, CUDA-synchronized, excluding dataloading and validation |
| Mean dataloading wait | `<...>` s (excluded from step time, logged for transparency) |
| Crop / batch / AMP | `<...>` / `<...>` / `<...>` |
| GPU | `<...>` |
| torch / CUDA / driver | `<...>` / `<...>` / `<...>` |
```

Supervision cost (from `results/supervision_cost.csv`, all measured in one
session on one GPU):

```markdown
| Supervision | Loss terms | Step time (s) | Relative | Peak alloc (GiB) | Relative |
|---|---|---|---|---|---|
| last_layer | 1 | | 1.00× | | 1.00× |
| deep_supervision | 4 | | | | |
| mutation (EMCAD default) | 15 | | | | |
```

### E. Ablation (optional)

```markdown
| Run | Loss | mIoU (%) | Small/thin group IoU (%) | Large group IoU (%) | Step time (s) |
|---|---|---|---|---|---|
| E0 primary | 0.3·CE + 0.7·Dice | | | | |
| E1 baseline | CE only | | | | |
```

Add one sentence naming the largest per-class deltas in both directions, and a
caveat that a single seed per arm does not establish significance.

---

## 25. Research summary outline

`RESEARCH_SUMMARY.md`, two to three pages. Write it for a reader who knows
EMCAD better than you do.

**1. Objective (3–4 sentences).** What was asked, what you built, the headline
result. Report what you measured; do not compare against published Cityscapes
numbers from methods trained with vastly more compute. *Figures: none.*

**2. Architecture adaptation (½ page).** The §5 finding: `num_classes` is
consumed at exactly four lines, the four heads change by 10,280 parameters,
`lib/` is byte-identical, verified by `diff`. Then the pipeline work that was
the real content. Do not overclaim novelty — the finding is that the codebase
generalized and you proved it, which is the right claim. *Table: the
architecture-vs-pipeline table from §5.*

**3. Correctness measures (½ page).** The two loss defects, with the measured
delta from `tests/test_losses.py`; the metric unit tests; the dataset suite; the
tiny-overfit result. Do not present these as heroics — present them as what
makes the numbers trustworthy. This section is the strongest evidence that you
can debug independently, which is one of the professor's stated criteria.
*Table: optional, one row per gate with its pass criterion.*

**4. Experimental setup (¼ page).** Table A from §24, compressed. Every
deviation from EMCAD defaults named: supervision mode, poly LR reinstated, AMP,
crop, iteration schedule. Do not bury them. *Table A.*

**5. Quantitative results (½ page).** mIoU, the per-class table, efficiency.
Do not describe a mid-range mIoU as competitive, and do not describe it as
poor — describe it as a baseline produced under a stated compute budget. *Tables
B, C, D.*

**6. Failure analysis (½–¾ page).** The thin-vs-large comparison with the
precision/recall distinction, top confusions, and the frequency relationship.
Hedged language throughout. Include the metric-geometry caveat from §20
unprompted. *Figures: one qualitative panel, `iou_vs_frequency.png`; Table: the
group summary.*

**7. Limitations (¼ page, do not skip).** Short schedule, half-resolution
training, single seed, deviation from mutation supervision, no multi-scale
inference, val-resize protocol if you used one, resume non-determinism. A
limitations section that names the things a reviewer would otherwise raise
reads as competence; its absence reads as inexperience.

**8. What more compute would buy (¼ page).** Ranked and specific: mutation
supervision at full cost; full 1024×2048 training; a longer schedule; multiple
seeds for significance; boundary-IoU analysis to separate metric geometry from
model failure. Each with one line on what it would resolve. Frame these as
experiments with questions attached, not a wish list — that is the difference
between sounding like a student and sounding like a researcher.

Throughout: label every number as measured, and every explanation as a
hypothesis.

---

## 26. Interview preparation

Likely questions, each with the concept you must hold and the shape of a strong
answer.

**1. Why was no major architecture rewrite needed?**
*Concept:* separation between task-dependent output projection and
task-independent feature extraction.
*Answer:* `num_classes` is consumed at exactly four lines, 83–86, as the
`out_channels` of the four 1×1 heads; the encoder and the decoder never see it.
I verified by grep, by a forward pass printing all four shapes, and by measuring
that the heads hold 19,532 of 26.8M parameters.

**2. How did `num_classes` propagate?**
*Concept:* reading an unfamiliar codebase by tracing a value.
*Answer:* `train_synapse.py:24` → `EMCADNet.__init__` (line 11) → lines 83–86
only. `EMCAD.__init__` (`decoders.py:316`) has no class argument. Nothing in
`forward` references it.

**3. Why modify the loss handling?**
*Concept:* ignore semantics are per-component, not global.
*Answer:* `CrossEntropyLoss()` at `trainer.py:68` has no `ignore_index` and
raises on 255. And `DiceLoss._dice_loss` includes ignored pixels in `z_sum`
while excluding them from the numerator; I demonstrated it by perturbing only
ignored-region logits — the official loss moved 0.0146 and leaked gradient, the
masked version moved 0.0.

**4. Why `ignore_index=255`?**
*Concept:* Cityscapes evaluation protocol.
*Answer:* 15 of the 34 labelIds are `ignoreInEval`; the official
`labelTrainIds` maps them to 255. They must not contribute to loss or IoU, or
the metric is not comparable to anyone else's. The ego-vehicle region alone is a
large coherent block in every frame.

**5. Why trainIds rather than labelIds?**
*Concept:* label-space semantics.
*Answer:* labelIds is the 34-class annotation space; trainIds is the 19-class
benchmark space with everything else at 255. I used the official masks where
present; where a mirror lacked them I remapped with the official table, which
my code cross-checks against `cityscapesscripts` and refuses to run on
disagreement.

**6. Why that supervision mode?**
*Concept:* cost scaling with resolution and class count. See §6.
*Answer:* the four-part answer in §6, with your own
`supervision_cost.csv` numbers.

**7. What is mutation supervision?**
*Concept:* the powerset construction in `trainer.py:91-100`.
*Answer:* all 15 non-empty subsets of the four heads; within a subset the
**logits are summed** and CE+Dice applied to that sum, so every additive
combination of scales must be independently decodable — an implicit ensemble
regularizer. (If you cannot explain *why logits are summed rather than losses
averaged*, you have not understood it. Be ready for that follow-up.)

**8. Why are small classes difficult?**
*Concept:* loss signal proportional to pixel count; IoU sensitivity inverse to
object area; feature stride.
*Answer:* three compounding reasons, with your frequency table and the
precision/recall split as evidence, plus the honest note that part of the gap is
metric geometry rather than model failure.

**9. Why might Dice help?**
*Concept:* per-class normalization.
*Answer:* Dice is computed per class and then averaged, so a class contributes
independently of its pixel count — structurally closer to what mIoU rewards
than CE's mean pixel likelihood.

**10. Why might Dice hurt?**
*Concept:* gradient behaviour on rare and absent classes.
*Answer:* it is noisy for classes with few pixels in a crop; absent classes
force a policy choice (skip or include) that changes the loss scale from batch
to batch; and the gradient can be unstable early when predictions are near
uniform. Which is why CE alone was the baseline and the pipeline was validated
on CE before Dice was added.

**11. How did you validate mIoU?**
*Concept:* testing a metric before trusting it.
*Answer:* eight unit tests including perfect prediction, all-wrong, a two-class
swap that must degrade exactly two IoUs, an all-ignore target, ignore-invariance,
absent-class NaN handling, a hand-computed 4-pixel case giving exactly 1/3, and
an accumulation test. `<plus the official cross-check result if you ran it>`.

**12. How did you profile memory?**
*Concept:* allocated vs reserved vs `nvidia-smi`.
*Answer:* `reset_peak_memory_stats()` before the window,
`max_memory_allocated()` after a synchronize; reported allocated as the headline
with reserved alongside. `nvidia-smi` would include the caching allocator and
the CUDA context.

**13. Why synchronize CUDA for timing?**
*Concept:* asynchronous kernel launch.
*Answer:* launches return immediately, so an unsynchronized CPU timer measures
launch overhead. I synchronize on both sides and discard 20 warm-up iterations
covering context creation, cuDNN algorithm selection and allocator growth.

**14. How does crop size affect training?**
*Concept:* context, batch statistics, memory, and what the metric sees.
*Answer:* a smaller crop reduces context, which matters most for large
contiguous classes; memory scales roughly linearly in pixels, so crop trades
against batch size; and if validation is also resized, thin structures are
smaller at evaluation time, which affects thin-class IoU specifically. I trained
at 512×1024 for memory and evaluated at `<your protocol>`, and I state it.

**15. What limitations remain?**
Have five ready, unprompted: short schedule, half-resolution training, single
seed, supervision deviation, and the evaluation-resolution protocol.

**16. What would you do with more compute?**
Ranked with the question each answers: mutation supervision at full cost; full
1024×2048 training; longer schedule; multiple seeds; boundary IoU to separate
metric geometry from model failure.

**Two more you should expect**, because they probe judgement rather than recall:

**17. "Your mIoU is well below published Cityscapes numbers. Why?"**
Do not get defensive and do not blame the model. Short schedule, half
resolution, no multi-scale inference, no tuning, a decoder designed for medical
segmentation applied unchanged. The goal was a correct and reproducible
baseline, not a competitive number — and I can tell you exactly which of those
factors I would remove first and why.

**18. "What surprised you?"**
Have a real answer. A good one: that the architecture needed no change at all,
and that the genuinely hard part was ignore-index correctness across three
independent components — the Dice defect was invisible in the loss curve and
only showed up when I wrote a test that perturbed ignored pixels.

---

## 27. Troubleshooting tree

Diagnose before fixing. Each entry: the symptom, the diagnostic to run, then the
smallest safe fix.

**Model import failure** (`ModuleNotFoundError: lib.networks`)
*Diagnose:* `pwd; ls lib/` — are you in the repo root with `lib/` present?
*Fix:* `cd` to the repo root; `cp -r /kaggle/working/EMCAD/lib .`. The scripts
insert their own directory into `sys.path`, so running from elsewhere is the
usual cause.

**timm import failure** (`cannot import name 'trunc_normal_tf_'`)
*Diagnose:* `python -c "import timm; print(timm.__version__)"`; then
`python -c "from timm.models.layers import trunc_normal_tf_"`.
*Fix:* `pip install -U timm`. Deprecation *warnings* from
`timm.models.helpers` are harmless; only an ImportError matters.

**Pretrained weights failure**
*Diagnose:* `ls -la pretrained_pth/pvt/`; `python -c "import torch;
print(type(torch.load('pretrained_pth/pvt/pvt_v2_b2.pth',
weights_only=False)))"`.
*Fix (torch ≥ 2.6):* `lib/networks.py:68` → `torch.load(path,
weights_only=False)`. The notebook patches this automatically. *Fix (file
missing):* attach a dataset containing it, or `--no-pretrain` and declare it —
expect a materially lower mIoU, which must then be stated in the README.
*Silent failure to watch for:* `networks.py:70` filters the state dict by key
intersection, so a mismatched checkpoint loads **zero** weights without error.
Verify by comparing a parameter before and after construction, or check that
your M4 overfit converges in a few hundred iterations rather than thousands.

**mmcv error**
*Diagnose:* `grep -rn "mmcv" lib/ *.py` — this project imports nothing from
mmcv.
*Fix:* EMCAD's README suggests installing mmcv-full for its own setup; it is not
needed for this code path. Skip it. If something else pulled it in, remove that
dependency rather than fighting the install, which is slow and version-fragile.

**CUDA mismatch** (`CUDA error: no kernel image is available`)
*Diagnose:* `python -c "import torch; print(torch.__version__,
torch.version.cuda, torch.cuda.get_device_capability())"`; `nvidia-smi`.
*Fix:* on Kaggle, use the preinstalled torch rather than installing your own
build. Reinstalling torch in a Kaggle session is the most common way to break
CUDA there.

**Out of memory**
*Diagnose:* note where it happens — model construction (impossible at this
size), forward, backward, or validation. Run `python -m tools.supervision_cost`
to see the per-mode cost.
*Fix, in order of preference:* 1) `--supervision last_layer` (biggest single
saving); 2) `--batch-size 1 --grad-accum 2`; 3) `--amp` if off; 4) crop 512×512;
5) `--val-resize 512 1024` if validation specifically OOMs (it runs at batch 1
with no graph, so this is rarely the cause). Record whatever you changed — it
goes in the efficiency table.

**Labels outside range** (`assert target out of range` from the metric)
*Diagnose:* `python -c "
import numpy as np; from PIL import Image
print(np.unique(np.array(Image.open('<a mask>'))))"`
*Fix:* if you see 19–33, you are reading `labelIds` → run
`tools/make_train_ids.py`. If you see 19–254, bilinear interpolation touched the
mask → find the transform; `tests/test_dataset.py` t7 localizes it.

**CUDA device-side assert**
*Diagnose:* re-run with `CUDA_LAUNCH_BLOCKING=1` to get a truthful traceback;
the message is otherwise attributed to a later, unrelated operation.
*Fix:* almost always a class index outside `[0, C-1]` reaching CE without
`ignore_index`, i.e. a 255. Check `CombinedSegLoss` was constructed with
`ignore_index=255`, and run the label-range diagnostic above.

**NaN loss**
*Diagnose:* re-run the same iteration with `--no-amp`. If NaN disappears, it is
fp16 overflow; if it persists, it is the loss or the data.
*Fix:* AMP overflow → the `GradScaler` normally handles it, so persistent NaN
under AMP suggests an inf in the forward pass; check the input normalization.
Loss-side → the masked Dice cannot divide by zero (smooth in numerator and
denominator) so suspect CE with a bad label. Also check LR: `train_cityscapes.py`
raises with an explicit message on the first non-finite loss rather than letting
it poison the weights silently.

**Tensor shape mismatch** (`size of tensor a (X) must match tensor b (X-1)`)
*Diagnose:* the error comes from the decoder's additive aggregation.
*Fix:* your crop is not divisible by 32 (§0.1). The dataset asserts this at
construction; if you bypassed it, do not.

**Dice loss not decreasing while CE does**
*Diagnose:* `python -m tests.test_losses` to confirm the implementation; then
check how many classes are present per crop and whether `skip_absent_classes`
is on.
*Fix:* this is often not a bug — batch-level Dice over 19 classes on a 512×1024
crop is dominated by whichever classes happen to be present, so it is noisier
than CE by nature. Judge it over hundreds of iterations, not tens. If it is
genuinely flat while CE falls, try `skip_absent_classes=False` to see whether
the policy is the cause, and report what you found.

**Tiny-overfit failure** — see the table in §12. Do not proceed past M4.

**Low mIoU after full training**
*Diagnose, in this order:* 1) did M4 pass? If not, the bug is upstream. 2) Does
training-time val mIoU match `eval_cityscapes.py`? A gap means a resize or
head-selection mismatch. 3) Is per-class IoU uniformly low, or only thin
classes? Uniformly low points at optimization (LR, schedule, pretrained weights
not loaded); thin-only is the expected pattern. 4) Check `n_valid_classes == 19`.
*Fix:* treat "train longer" as the last hypothesis, not the first.

**Incorrect visualization** (colours wrong, mask misaligned, panels disagree
with metrics)
*Diagnose:* confirm `visualize_cityscapes.py` and `eval_cityscapes.py` used the
same checkpoint and the same `--val-resize` — both are recorded in
`qualitative_manifest.json` and `summary.json`.
*Fix:* if the palette looks wrong, you are probably rendering labelIds with the
trainId palette. If the mask is misaligned, go back to M2 — the metrics are
wrong too, you just cannot see it in a number.

**Metric mismatch with official Cityscapes evaluation** — §14. Check, in order:
resolution (most likely), ignore handling, class ordering. A gap confined to one
or two classes is the trainId→labelId map for those classes.

---

## 28. Master checklist

Ordered, with a completion criterion for every item. Nothing here is a date.

### Repository and environment
- [ ] New repository created (not a fork); `lib/` vendored — *criterion:* `diff -r lib /tmp/EMCAD_ref/lib` is empty or shows only the `weights_only` line
- [ ] Upstream commit recorded — *criterion:* `results/emcad_commit.txt` exists
- [ ] Attribution in README, NOTICE, LICENSE, file headers — *criterion:* EMCAD, the CVPR 2024 paper and the repo link appear in the README's first screen
- [ ] Kaggle session configured — *criterion:* `tools/check_env.py` reports a real GPU
- [ ] M0 passed — *criterion:* `results/environment.json` written, all checks PASS
- [ ] Pretrained weights hashed — *criterion:* `results/pretrained_sha256.txt` exists

### Code understanding
- [ ] `lib/networks.py` read in full — *criterion:* you can state, without looking, where `num_classes` is used
- [ ] `EMCAD.forward` traced — *criterion:* you can write the shape after each of the 16 operations
- [ ] `trainer.py:91-113` understood — *criterion:* you can explain why `iout` sums logits
- [ ] `utils/utils.py:102-138` understood — *criterion:* you can derive the ignore defect on paper

### Model smoke test
- [ ] M1 passed at the real crop — *criterion:* four `(B,19,512,1024)` shapes, CE ≈ 11.78, finite grad norm, `MODEL SMOKE TEST PASSED`
- [ ] Divisibility behaviour confirmed — *criterion:* the non-divisible input raises a `RuntimeError`

### Dataset
- [ ] Data located; mask source known — *criterion:* `mask_kind` printed and recorded in the README
- [ ] trainIds present or generated — *criterion:* 2975 + 500 `labelTrainIds` files
- [ ] M2 automated checks passed — *criterion:* 10/10, saved to `logs/dataset_validation.log`
- [ ] Overlays inspected **by eye** — *criterion:* you have looked at ≥ 8 and confirmed alignment
- [ ] Class frequency computed — *criterion:* `results/class_frequency.csv`, all 19 non-zero

### Transforms
- [ ] Geometric ops synchronized — *criterion:* M2 t6 passed and overlays are aligned
- [ ] Nearest-only on masks — *criterion:* M2 t7 passed over 200 augmented samples
- [ ] Crop padding uses 255 for masks — *criterion:* confirmed by reading `_joint_transform`

### Loss
- [ ] CE has `ignore_index=255` — *criterion:* `tests/test_losses.py` green
- [ ] Masked Dice implemented — *criterion:* ignore-invariance and zero-gradient tests pass
- [ ] The defect demonstrated — *criterion:* the official-vs-masked delta is in `logs/`
- [ ] Supervision subsets reproduce the repo — *criterion:* 15 / 4 / 1 asserted

### Metrics
- [ ] M5 passed — *criterion:* `ALL METRIC TESTS PASSED` saved to `logs/`
- [ ] Absent-class policy decided and documented — *criterion:* stated in the README

### Tiny overfit
- [ ] M4 passed — *criterion:* `logs/overfit.json` has `"passed": true`
- [ ] Subset and scored classes recorded — *criterion:* image names and class list in that JSON

### Profiling
- [ ] M6 passed — *criterion:* `results/supervision_cost.csv` with three rows
- [ ] Supervision mode chosen from measurement — *criterion:* the justification cites your own numbers
- [ ] Step-time definition written down — *criterion:* the §15 sentence is in the README verbatim

### Short training
- [ ] M7 passed — *criterion:* loss falls and val mIoU is clearly above chance over 300 iterations
- [ ] Time budget arithmetic done — *criterion:* measured step time × `max_iters` fits with slack

### Full training
- [ ] M8 complete — *criterion:* `max_iters` reached; `best_miou.pth` and `train_summary.json` exist
- [ ] Nothing changed mid-run — *criterion:* one `config.json` for the whole run directory
- [ ] Logs committed — *criterion:* `train_log.csv` in git

### Evaluation
- [ ] M9 complete — *criterion:* `results/per_class_iou.csv` + `summary.json`, `n_valid_classes == 19`
- [ ] Evaluation protocol stated — *criterion:* native or resized, written in the README
- [ ] Official cross-check run or explicitly deferred — *criterion:* either the comparison or a sentence saying why not

### Visualizations
- [ ] M10 figures produced — *criterion:* ≥ 6 five-panel PNGs plus the manifest
- [ ] Selection strategy declared — *criterion:* `spread` (or stated otherwise) in the manifest and the README
- [ ] Figures match the metrics — *criterion:* same checkpoint and resize in both JSONs

### Failure analysis
- [ ] Quantitative half produced — *criterion:* `failure_analysis.json`, `thin_vs_large.csv`, both figures
- [ ] Eleven conditions inspected — *criterion:* a record for each, with evidence
- [ ] Every claim paired with a number — *criterion:* no visual claim stands alone
- [ ] Hedged language throughout — *criterion:* no causal assertion without an experiment behind it
- [ ] Metric-geometry caveat included — *criterion:* present in §20's write-up

### Results
- [ ] Tables A–D filled — *criterion:* zero placeholders
- [ ] Ablation table filled or omitted — *criterion:* no half-filled table
- [ ] Every number traceable to a file — *criterion:* each table cites its source file

### README
- [ ] All 23 required items from the brief present — *criterion:* checked off one by one
- [ ] Deviations declared — *criterion:* supervision, poly LR, AMP, crop, schedule, any `weights_only` edit
- [ ] Exact commands reproduce the run — *criterion:* copy-paste works in a clean clone
- [ ] Reproducibility caveats stated — *criterion:* the §23 paragraph is present

### Research summary
- [ ] `RESEARCH_SUMMARY.md` written — *criterion:* all eight §25 sections
- [ ] Limitations section present — *criterion:* at least five named
- [ ] Nothing overclaimed — *criterion:* every causal statement is hedged or backed by an experiment

### Interview preparation
- [ ] All eighteen §26 questions answerable aloud — *criterion:* without notes
- [ ] `EMCAD.forward` drawable from memory — *criterion:* on a blank page, with shapes
- [ ] Mutation supervision explainable — *criterion:* including why logits are summed
- [ ] Supervision deviation defence rehearsed — *criterion:* the four-part answer with your measured numbers
- [ ] One honest "what surprised me" answer — *criterion:* specific, not generic
