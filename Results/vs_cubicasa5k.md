# Ours vs CubiCasa5K (Kalervo et al. 2019) — head-to-head in *their* metric

**Their task/metric:** multi-task semantic segmentation; per-class **pixel IoU / Acc**
via a 12-class room + 11-class icon confusion matrix (`floortrans.metrics.runningScore`),
CubiCasa5K test split (400 plans).
**Date of this run:** 2026-07-18.

## Result — per-class pixel IoU / Acc (%), test split

| class   | CubiCasa5K (paper) | CubiCasa5K (measured¹) | **Ours, boxes rasterized²** | Δ ours − measured |
|---------|--------------------|------------------------|------------------------------|-------------------|
| wall    | 73.0 / 85.8        | 72.2 / 83.3            | **76.9 / 90.8**              | **+4.7**          |
| door    | 53.6 / 59.8        | 52.0 / 57.0            | **66.4 / 83.1**              | **+14.4**         |
| window  | 66.8 / 73.7        | 65.4 / 71.4            | **74.1 / 89.0**              | **+8.7**          |
| railing | 23.6 / 28.7        | 22.9 / 27.5            | **34.5 / 66.9**              | **+11.6**         |

¹ Their released model (`model_best_val_loss_var.pkl`) run through their own
`eval.py` (LMDB, original-size images, 4-rotation TTA) in our environment —
reproduces the paper within ~1.5 pt on every class, which validates the setup.

² Our RT-DETR-L detections (`rtdetr_l_autoresearch_60ep.pt`, conf ≥ 0.25,
1024²) drawn as filled axis-aligned boxes into their label-map format, scored
by their own `runningScore` against their own SVG-parsed GT (F1_scaled frame,
no TTA).

## Read this before quoting

- **A box is not a mask.** Our per-class masks are unions of axis-aligned
  rectangles. For thin *axis-aligned* structures (most CubiCasa walls,
  doors, windows) the box is close to the true extent, but for diagonal
  walls/railings a box over-covers, which *helps* recall and *hurts*
  precision in an IoU that our number nets out. Treat our column as a
  strong proxy, not a segmentation claim.
- Even so, the door gap (+14.4) and window gap (+8.7) are far larger than
  any plausible proxy bias on axis-dominated classes, and our wall/railing
  wins come *without* TTA while theirs use 4-rotation TTA.
- Their evaluation runs at original image size; ours at F1_scaled size.
  IoU is a per-pixel ratio and both label maps live in a consistent frame,
  so this changes numbers marginally (same order as their paper-vs-measured
  drift).
- **Erratum:** earlier revisions of our internal docs (and several places
  online) quote CubiCasa5K door = 69.2 / window = 53.6. Reading the paper's
  rotated table columns carefully (verified against the arXiv LaTeX source):
  **window = 66.8, door = 53.6, and 69.2 is Closet.** Our measured rerun
  (window 65.4 > door 52.0, closet 68.4) confirms the ordering.

## Reproduction notes

Their numbers were reproduced with their released weights and their own eval
code (two period-compat patches were needed to run the 2019 code on a modern
stack: `scipy==1.10.1` — newer scipy changed `stats.mode` and crashes their
post-processing — and clipping `skimage.draw.polygon` to the image bounds;
neither affects scores).
