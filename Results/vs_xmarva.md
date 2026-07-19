# Ours vs xmarva/floorplan-detection — identical-protocol head-to-head

**Their setup:** Cascade R-CNN (ResNet-101 + FPN, mmdetection 3.x), trained on
their own COCO-bbox conversion of CubiCasa5K, classes **wall + room** only,
test-time resize 1333×800. **No metrics are published in the repo**, so we ran
their released checkpoint (`cascade_swin_latest.pth`, Google Drive) on the same
400 test floor plans and scored *both* models with the *same* three protocols —
the only fully same-eval comparison in this study.
**Date of this run:** 2026-07-18 (mmdet 3.3.0).

## Wall — same images, same GT, same scorer (%)

| protocol (identical for both) | **Ours RT-DETR-L** (autoresearch-60ep) | xmarva Cascade R-CNN |
|---|---|---|
| standard COCO bbox AP50            | **82.3** | 29.0 |
| standard COCO bbox AP@[.5:.95]     | **52.8** | 14.6 |
| canonical polygon-GT AP@0.5        | **73.3** | 26.9 |
| box-raster pixel IoU (conf ≥ 0.25) | **77.5** | 27.2 |

## Fairness verification (why 29.0 is real, not our harness being hostile)

Their wall boxes might have used a different GT convention, so we also scored
their predictions against **their own** `test_coco_pt.json` annotations
(F1_original coordinate frame, their wall/room definitions, predictions
rescaled per image):

| class (their GT, their convention) | AP50 | AP75 | AP@[.5:.95] |
|---|---|---|---|
| wall | 29.3 | 12.3 | 14.3 |
| room | **88.7** | 78.7 | 74.2 |

Wall AP is ~29 under both GT conventions — the two harnesses cross-validate —
and the model is genuinely strong on **room** detection (88.7 AP50), a class we
do not model. The released checkpoint is simply room-focused: walls are thin,
dense (≈27 per plan), and evaluated after a 1333×800 downscale, and their
training recipe did not solve that regime.

## Caveats

- Their repo targets an API demo, not a benchmark; the author published the
  checkpoint without claiming numbers. This comparison should be read as
  "our measurement of their released artifact", not as a refutation of a
  published result.
- Their model saw F1_original images in training; we feed F1_scaled (both are
  the same drawings; mmdetection's resize to 1333×800 makes the inputs nearly
  identical, and their room AP50 of 88.7 on our images confirms no
  domain-shift artifact).
- Door / window / railing: not in their class set — no comparison possible.
- Their two other released checkpoints (faster_rcnn, retinanet) were not
  evaluated; cascade_swin is the one their README recommends.
