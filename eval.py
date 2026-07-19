#!/usr/bin/env python
"""Evaluate a checkpoint on the CubiCasa5K test split under three protocols.

  1. standard COCO bbox AP (pycocotools; the protocol MuraNet/YOLO papers use)
  2. polygon-GT AP (axis-aligned box predictions vs polygon GT, Shapely IoU —
     ~0.1-0.15 stricter than COCO AP; our canonical reporting protocol)
  3. box-rasterized pixel IoU (proxy for segmentation-IoU comparisons)

The dataset is NOT included in this release. You need COCO-style annotations
with polygon segmentations (categories: 1=wall 2=door 3=window 5=railing
6=linkage_point) and the referenced images:

    CubiCasaPath/
    ├── annotations/test.json        # COCO polygons
    └── <file_name paths from the json>

Usage:
    python eval.py --model weights/rtdetr_l_autoresearch_60ep.pt \
        --coco-json /path/to/CubiCasaPath/annotations/test.json \
        --img-root  /path/to/CubiCasaPath [--max-images 5]

Part of the Architect Ant paper: https://arxiv.org/abs/2606.10953
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

CLS_NAMES = ["wall", "door", "window", "railing", "linkage_point"]
CLS_TO_CAT = {0: 1, 1: 2, 2: 3, 3: 5, 4: 6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/rtdetr_l_autoresearch_60ep.pt")
    ap.add_argument("--coco-json", required=True,
                    help="COCO annotations with polygon segmentations, e.g. "
                         "CubiCasaPath/annotations/test.json")
    ap.add_argument("--img-root", required=True,
                    help="root the json's file_name paths are relative to")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="threshold for polygon-AP / pixel-IoU (canonical 0.25)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from PIL import Image, ImageDraw
    from ultralytics import RTDETR
    from polygon_eval import evaluate_polygon_ap

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.coco_json) as f:
        gt_raw = json.load(f)
    if args.max_images > 0:
        keep = {im["id"] for im in gt_raw["images"][: args.max_images]}
        gt_raw["images"] = [im for im in gt_raw["images"] if im["id"] in keep]
        gt_raw["annotations"] = [a for a in gt_raw["annotations"]
                                 if a["image_id"] in keep]

    cat_to_cls = {v: k for k, v in CLS_TO_CAT.items()}
    gt_polys = defaultdict(lambda: defaultdict(list))
    for a in gt_raw["annotations"]:
        if a["category_id"] in cat_to_cls and a.get("segmentation"):
            poly = np.asarray(a["segmentation"][0], float).reshape(-1, 2)
            if len(poly) >= 3:
                gt_polys[a["image_id"]][cat_to_cls[a["category_id"]]].append(poly)

    model = RTDETR(args.model)
    model.to(device)
    print(f"[eval] {Path(args.model).name} on {device}, "
          f"{len(gt_raw['images'])} images, imgsz={args.imgsz}")

    dets, poly_preds, poly_gts = [], [], []
    inter = np.zeros(len(CLS_NAMES))
    union = np.zeros(len(CLS_NAMES))
    img_root = Path(args.img_root)

    for n, im in enumerate(gt_raw["images"]):
        p = img_root / im["file_name"]
        if not p.exists():
            print(f"[eval] MISSING {p}", file=sys.stderr)
            continue
        r = model.predict(str(p), conf=0.001, iou=0.5, imgsz=args.imgsz,
                          max_det=300, verbose=False, device=device)[0]
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy() if b is not None else np.zeros((0, 4))
        cls = b.cls.cpu().numpy().astype(int) if b is not None else np.zeros(0, int)
        conf = b.conf.cpu().numpy() if b is not None else np.zeros(0)

        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
            dets.append({"image_id": im["id"], "category_id": CLS_TO_CAT[int(c)],
                         "bbox": [float(x1), float(y1),
                                  float(x2 - x1), float(y2 - y1)],
                         "score": float(s)})

        keep = conf >= args.conf
        pp = {"polygons": [], "labels": [], "scores": []}
        for (x1, y1, x2, y2), c, s in zip(xyxy[keep], cls[keep], conf[keep]):
            pp["polygons"].append(np.array([[x1, y1], [x2, y1],
                                            [x2, y2], [x1, y2]], float))
            pp["labels"].append(int(c))
            pp["scores"].append(float(s))
        gg = {"polygons": [], "labels": []}
        for ci in range(len(CLS_NAMES)):
            for poly in gt_polys[im["id"]].get(ci, []):
                gg["polygons"].append(poly)
                gg["labels"].append(ci)
        poly_preds.append(pp)
        poly_gts.append(gg)

        W, H = im["width"], im["height"]
        for ci in range(len(CLS_NAMES)):
            pm = Image.new("1", (W, H), 0)
            d = ImageDraw.Draw(pm)
            for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
                if int(c) == ci and s >= args.conf:
                    d.rectangle([x1, y1, x2, y2], fill=1)
            gm = Image.new("1", (W, H), 0)
            g = ImageDraw.Draw(gm)
            for poly in gt_polys[im["id"]].get(ci, []):
                g.polygon([tuple(q) for q in poly], fill=1)
            pa, ga = np.asarray(pm, bool), np.asarray(gm, bool)
            inter[ci] += np.logical_and(pa, ga).sum()
            union[ci] += np.logical_or(pa, ga).sum()
        if (n + 1) % 50 == 0:
            print(f"[eval] {n+1}/{len(gt_raw['images'])}")

    if not dets:
        sys.exit("[eval] no detections — check paths")

    tmp = Path(".gt_tmp.json")
    with open(tmp, "w") as f:
        json.dump(gt_raw, f)
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(tmp))
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = list(CLS_TO_CAT.values())
        ev.evaluate(); ev.accumulate()
    tmp.unlink(missing_ok=True)

    def class_ap(cat_id, thr=None):
        pr = ev.eval["precision"][:, :, ev.params.catIds.index(cat_id), 0, -1]
        if thr is not None:
            t = int(np.argmin(np.abs(ev.params.iouThrs - thr)))
            pr = pr[t:t + 1]
        pr = pr[pr > -1]
        return float(pr.mean()) if pr.size else float("nan")

    pap = evaluate_polygon_ap(poly_preds, poly_gts, iou_thresholds=[0.5, 0.75])

    per_class = {}
    for ci, name in enumerate(CLS_NAMES):
        cat = CLS_TO_CAT[ci]
        per_class[name] = {
            "coco_AP@0.5": class_ap(cat, 0.5),
            "coco_AP@0.75": class_ap(cat, 0.75),
            "coco_AP@[.5:.95]": class_ap(cat),
            "polygon_AP@0.5": float(pap["AP@0.5"]["per_class"].get(ci, 0.0)),
            "polygon_AP@0.75": float(pap["AP@0.75"]["per_class"].get(ci, 0.0)),
            "pixel_iou_boxraster": (float(inter[ci] / union[ci])
                                    if union[ci] else float("nan")),
        }
    out = {
        "model": args.model, "imgsz": args.imgsz, "conf": args.conf,
        "num_images": len(gt_raw["images"]),
        "coco_mAP@0.5": float(np.mean([v["coco_AP@0.5"] for v in per_class.values()])),
        "polygon_mAP@0.5": float(np.mean([v["polygon_AP@0.5"] for v in per_class.values()])),
        "per_class": per_class,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
