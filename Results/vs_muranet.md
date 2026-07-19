# Ours vs MuraNet (arXiv:2309.00348, 2023) — head-to-head in *their* metrics

**Their task/metrics:** multi-task — door/window **bbox detection** scored with
standard COCO AP (AP50 and AP@[.5:.95]) + wall **segmentation IoU**, evaluated
at **1536×1536**.
**Their code/weights are not released**, so their column is paper numbers
(verbatim from the arXiv LaTeX source, `arXiv-2309.00348v1/main.tex`, Tables 1–2);
our column is measured on the **official test split** under the same COCO
protocol (pycocotools, bbox GT), at our native **1024²**.
**Date of this run:** 2026-07-18 (split analysis added 2026-07-19).

> ⚠️ **Split warning (found 2026-07-19).** MuraNet does **not** use the
> official CubiCasa5K 4200/400/400 split — their §4.1: *"4000 images for
> training, 500 for validation, and 500 for testing."* Their evaluation
> images are a custom sample, every accuracy figure in the paper is captioned
> "*Validation* …", and the result tables never say which of their sets the
> numbers come from. If their split was random, ~80 % of the *official* test
> images were inside *their training set*. Additionally they train for
> **1000 epochs** (50-epoch warmup) vs our 60. Measured with our own model on
> the official splits, split choice alone moves per-class AP50 by up to
> ±1.8 pt (door −1.7, wall +1.8 going test→val) — the same scale as
> MuraNet's entire +2.1 AP50 margin. Their AP50 lead is therefore **within
> the comparability noise floor**; our +6 AP@[.5:.95] lead is far outside it.

## Door / window detection — standard COCO bbox AP (%)

| model | input | AP50 door | AP50 window | AP50 mean | AP[.5:.95] door | AP[.5:.95] window | AP[.5:.95] mean |
|---|---|---|---|---|---|---|---|
| MuraNet base            | 1536² | **91.2** | **92.2** | **91.7** | 47.9 | 59.7 | 53.8 |
| MuraNet+spp             | 1536² | 90.9 | 91.5 | 91.2 | 47.7 | 59.1 | 53.4 |
| YOLOv3 base (their bl.) | 1536² | 89.2 | 90.1 | 89.6 | 43.6 | 55.4 | 49.5 |
| **Ours RT-DETR-L** (autoresearch-60ep) | 1024² | 89.8 | 89.3 | 89.6 | **57.5** | **62.3** | **59.9** |
| Ours RT-DETR-L, 1536-trained variant | 1536² | 90.4 | 88.7 | 89.6 | **58.4** | 62.5 | **60.4** |

**Reading:** at the loose IoU=0.5 threshold MuraNet leads by 2.1 pt mean while
we exactly tie their YOLOv3 baseline (89.6) — at two-thirds of their input
resolution. At the strict averaged threshold **we lead MuraNet by +6.1 pt**
(59.9 vs 53.8) and their YOLOv3 by +10.4: RT-DETR's boxes are markedly tighter
once localization quality is priced in. MuraNet finds slightly more openings
at IoU 0.5; ours localizes the ones it finds substantially better.

**Resolution check.** Two experiments:
1. *Inference-time upscaling degrades:* the 1024-trained model evaluated at
   1536² drops to 84.5 dw-AP50.
2. *Retraining at 1536² does not close the gap:* a full 60-epoch run at
   1536² (same schedule) scores **dw-AP50 89.6 — identical to the 1024
   model** (door 90.4, window 88.7; they cancel), with AP@[.5:.95] up
   slightly to **60.4** and small-object gains (railing 56.3, linkage 61.0).

Conclusion: **MuraNet's 2.1-pt AP50 lead is not a resolution artifact** — at
matched 1536² input we still measure 89.6. The remaining gap is
architecture/recipe (or their unverifiable harness); our strict-localization
lead at matched resolution is +6.6 (60.4 vs 53.8). Since 1536 costs 2.25×
inference compute for +0.6 AP@[.5:.95] and ±0 AP50, the 1024 model stays our
primary checkpoint.

## Wall — their metric is segmentation IoU (%), ours is detection

| model | wall |
|---|---|
| MuraNet base (seg IoU, paper) | **78.4** |
| U-Net base (their baseline)   | 65.5 |
| Ours — box-raster proxy IoU¹  | 77.5 |
| Ours — wall bbox AP50 (no MuraNet counterpart) | 82.3 |

¹ Predicted boxes (conf ≥ 0.25) rasterized as masks vs polygon GT — an
optimistic proxy for thin/diagonal walls, **not** true segmentation IoU.
MuraNet does not report wall *detection* AP, so the clean wall comparison
does not exist; the honest statement is "their true wall seg-IoU 78.4 vs our
box-proxy 77.5 — approximately parity, with the proxy caveat in their favor."

## Caveats

- MuraNet numbers could not be independently verified (no code, no weights,
  no per-image outputs); they are taken at face value from the paper.
- **Their split is custom (4000/500/500), not the official one, and possibly
  val-not-test** — see the split warning at the top. Their training budget is
  1000 epochs vs our 60.
- Class scope differs: MuraNet handles wall/door/window only; we additionally
  detect railing (AP50 55.0) and linkage_point (60.0), for which they report
  nothing.
- Their AP is computed by their own harness; ours by pycocotools. Both claim
  the standard COCO protocol; residual harness differences of a few tenths
  of a point are possible.
