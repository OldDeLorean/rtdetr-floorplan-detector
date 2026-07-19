#!/usr/bin/env python
"""Train the RT-DETR-L structural floor-plan detector.

This is the exact recipe that produced the released checkpoint
(`weights/rtdetr_l_autoresearch_60ep.pt`): RT-DETR-L from COCO-pretrained
weights, 60 epochs at 1024², AdamW (ultralytics auto-LR), 1-epoch warmup,
cosine schedule with final LR = 0.1 x lr0, vertical+horizontal flips,
reduced color jitter (floor plans are near-grayscale).

Setup:
  1. Edit `data.yaml` -> point `path:` at your YOLO-format dataset
     (e.g. CubiCasaPath/cubicasa5k_yolo — the dataset is not included).
  2. Adjust DEVICE / BATCH below to your hardware
     (batch 16 @ 1024² fits 2x40 GB GPUs; use 8 for a single 24 GB GPU).
  3. python train.py

Outputs runs/train/weights/best.pt — evaluate it with eval.py.

Part of the Architect Ant paper: https://arxiv.org/abs/2606.10953
"""
from pathlib import Path

# ------------------------------ knobs ------------------------------------
MODEL   = "rtdetr-l.pt"      # auto-downloaded COCO-pretrained init
DATA    = Path(__file__).resolve().parent / "data.yaml"
EPOCHS  = 60
IMGSZ   = 1024
BATCH   = 16
DEVICE  = "0"                # "0,1" for two GPUs, "cpu" to test the pipeline
WORKERS = 8
SEED    = 0

# schedule (the two knobs that beat the default recipe: warmup 3->1, lrf 0.01->0.1)
OPTIMIZER     = "auto"       # resolves to AdamW, lr0 ~= 0.00111
LRF           = 0.1
WARMUP_EPOCHS = 1.0
COS_LR        = True
WEIGHT_DECAY  = 0.0005

# augmentation (floor plans: no canonical up -> flipud; minimal color jitter)
HSV_H, HSV_S, HSV_V = 0.0, 0.0, 0.2
FLIPUD, FLIPLR      = 0.5, 0.5
TRANSLATE, SCALE    = 0.1, 0.5
MOSAIC, CLOSE_MOSAIC = 1.0, 10

# loss gains (ultralytics defaults)
BOX_GAIN, CLS_GAIN, DFL_GAIN = 7.5, 0.5, 1.5
# --------------------------------------------------------------------------


def main():
    from ultralytics import RTDETR
    model = RTDETR(MODEL)
    model.train(
        data=str(DATA), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        device=DEVICE, workers=WORKERS, seed=SEED, amp=True, patience=0,
        project="runs", name="train", exist_ok=True,
        optimizer=OPTIMIZER, lrf=LRF, warmup_epochs=WARMUP_EPOCHS,
        cos_lr=COS_LR, weight_decay=WEIGHT_DECAY,
        hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
        flipud=FLIPUD, fliplr=FLIPLR, translate=TRANSLATE, scale=SCALE,
        mosaic=MOSAIC, close_mosaic=CLOSE_MOSAIC,
        box=BOX_GAIN, cls=CLS_GAIN, dfl=DFL_GAIN,
    )
    print("\n[train] done -> runs/train/weights/best.pt"
          "\n[train] evaluate: python eval.py --model runs/train/weights/best.pt"
          " --coco-json /path/to/CubiCasaPath/annotations/test.json"
          " --img-root /path/to/CubiCasaPath")


if __name__ == "__main__":
    main()
