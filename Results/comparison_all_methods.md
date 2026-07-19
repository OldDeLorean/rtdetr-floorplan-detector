# CubiCasa5K structural elements — all four methods, unified metrics

Comparison of our **RT-DETR-L** structural detector against the three reference
works on the CubiCasa5K **test split** (400 plans, same 4200/400/400 split
everywhere), classes **wall / door / window / railing**. Run 2026-07-18.
Head-to-head details per method: [vs_cubicasa5k.md](vs_cubicasa5k.md) ·
[vs_muranet.md](vs_muranet.md) · [vs_xmarva.md](vs_xmarva.md).

![comparison chart](comparison_chart.png)

No single metric is natively reported by all four works, so the unified view
has two blocks, each computed identically for every method that can produce it
("—" = not applicable / not published).

## Block 1 — pixel IoU (%), the segmentation-style view

| class   | **Ours** (box-raster ‡) | CubiCasa5K (measured seg) | MuraNet (paper seg †) | xmarva (box-raster ‡) |
|---------|--------------------------|---------------------------|------------------------|------------------------|
| wall    | 76.9                     | 72.2                      | **78.4**               | 27.2                   |
| door    | **66.4**                 | 52.0                      | —                      | —                      |
| window  | **74.1**                 | 65.4                      | —                      | —                      |
| railing | **34.5**                 | 22.9                      | —                      | —                      |

## Block 2 — standard COCO bbox AP50 (%), the detection view

| class   | **Ours** (measured) | CubiCasa5K | MuraNet (paper †, 1536²) | xmarva (measured) |
|---------|---------------------|------------|---------------------------|--------------------|
| wall    | **82.3**            | —          | —                         | 29.0               |
| door    | 89.8                | —          | **91.2**                  | —                  |
| window  | 89.3                | —          | **92.2**                  | —                  |
| railing | **55.0**            | —          | —                         | —                  |
| door/window AP@[.5:.95] mean | **59.9** | — | 53.8                 | —                  |

† paper numbers (MuraNet released no code/weights; input 1536² vs our 1024²).
‡ axis-aligned predicted boxes rasterized as masks — an optimistic proxy for
thin/diagonal objects, **not** true segmentation IoU.
CubiCasa5K's column in Block 1 is *measured* by us with their released model,
their own eval code and TTA (reproduces their paper within ~1.5 pt/class).

## Our full model card (canonical polygon-GT protocol, strictest of all)

Our own 5-class numbers under the canonical evaluator (axis-aligned box
predictions vs **polygon** GT, Shapely IoU — measured ≈ 0.1–0.15 *stricter*
than the COCO bbox AP above; the two AP columns are not comparable to Block 2):

| class | AP@0.5 | AP@0.75 |
|---|---|---|
| wall | 0.733 | 0.475 |
| door | 0.848 | 0.609 |
| window | 0.855 | 0.651 |
| railing | 0.470 | 0.192 |
| linkage_point | 0.566 | 0.237 |
| **mAP (5-class)** | 0.694 | **0.433** |

No reference method models railing detection or linkage points at all —
breadth is ours alone.

## Which checkpoint is "Ours"

`weights/rtdetr_l_autoresearch_60ep.pt` — the autoresearch-best schedule
(warmup=1, lrf=0.1, cos_lr, flipud=0.5; optimizer auto→AdamW lr 0.00111)
retrained for a full 60 epochs on 2026-07-18. **Chosen over the former 60ep
baseline on 2026-07-18** because the priority classes decide: it wins
wall/door/window at both AP@0.5 and AP@0.75 on the canonical protocol
(wall AP75 +4.1, door AP75 +6.8) and wins **every** class under the COCO
protocol. Its only regression is polygon-GT railing (−2.5 AP50, the
rarest/noisiest class), which alone turns the 5-class canonical mAP@0.5 into
a −0.0013 "tie" (0.6941 vs 0.6954); the val curve was saturated after ep≈50,
so longer training was not warranted.

## Verdicts, one line each

- **vs CubiCasa5K (2019):** we beat their segmentation model *in their own
  metric* on all four classes (+4.7 wall, +14.4 door, +8.7 window, +11.6
  railing IoU pts vs their measured numbers), with the box≠mask proxy caveat.
- **vs MuraNet (2023):** they report +2.1 door/window AP50 — but on a custom
  split (possibly validation), with a 1000-epoch budget, unverifiable, and
  within the measured ±2 pt split-noise floor; we lead the stricter
  AP@[.5:.95] by **+6.1 pt** (far outside that floor) and tie their YOLOv3
  baseline at AP50 (89.6). Wall: their true seg-IoU 78.4 ≈ our proxy 77.5.
- **vs xmarva:** under a fully identical protocol we lead wall detection
  82.3 vs 29.0 AP50 (their released checkpoint is room-focused — 88.7 room
  AP50 on their own GT — a class we don't model).
- **Breadth & size:** only our model covers all five structural classes in
  one detector — 32.8 M params in a 66 MB FP16 inference checkpoint,
  **2.7× fewer params than xmarva's Cascade R-CNN** (88.3 M / 353 MB FP32
  weights; their released 712 MB file is half optimizer state). CubiCasa5K's
  hourglass is smaller (17.4 M) but segmentation-only — and we beat it in its
  own metric. MuraNet released no weights.

## Caveat discipline (read before quoting any row)

1. Polygon-GT AP (our canonical protocol) is ~0.1–0.15 stricter than bbox-GT
   COCO AP — never mix the two in one column; this file never does.
2. Box-raster IoU (‡) over-credits axis-aligned thin objects and over-covers
   diagonal ones; it is a proxy on both sides of Block 1 where marked.
3. Input resolution differs: ours 1024², MuraNet 1536², CubiCasa5K original
   size + TTA, xmarva 1333×800. The resolution difference does not explain
   MuraNet's AP50 lead (verified by a 1536² retrain — see `vs_muranet.md`).
4. MuraNet rows are unverifiable paper numbers — **and measured on a custom
   4000/500/500 split, not the official test set** (possibly their validation
   set; their paper's figures are all "Validation …" and the tables don't
   name the split; they also train 1000 epochs vs our 60). Split choice alone
   moves per-class AP50 by ~±2 pt (measured with our model, official
   val vs test), i.e. MuraNet's +2.1 AP50 margin sits inside the
   comparability noise. Every non-MuraNet number in this folder was measured
   by us.
5. Erratum: CubiCasa5K paper icon columns are frequently misquoted; correct
   test IoU is window 66.8 / door 53.6 (69.2 is Closet). Verified from the
   arXiv source and reproduced by measurement.
