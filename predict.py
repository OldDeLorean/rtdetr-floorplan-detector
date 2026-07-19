#!/usr/bin/env python
"""Structural floor-plan detection on arbitrary images.

Runs the RT-DETR-L detector (walls / doors / windows / railings / linkage
points, trained on CubiCasa5K) on one image or a whole folder, and writes an
annotated render (bold hatched boxes; walls at the bottom of the z-stack,
doors & windows on top) plus an optional JSON with the raw boxes.

Part of the Architect Ant paper: https://arxiv.org/abs/2606.10953

Usage:
    python predict.py path/to/plan.png
    python predict.py path/to/folder --out-dir out/ --save-json
    python predict.py plan.jpg --conf 0.15 --device cpu

Outputs <name>_pred.png (and <name>_pred.json with --save-json) next to each
input, or into --out-dir if given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLS = ["wall", "door", "window", "railing", "linkage_point"]
COLOR = {0: (42, 120, 214), 1: (0, 131, 0), 2: (213, 81, 129),
         3: (201, 133, 0), 4: (74, 58, 167)}
Z = {0: 0, 3: 50, 4: 50, 1: 100, 2: 100}   # walls bottom; doors/windows top
OUTLINE_W = 5
HATCH_STEP = 12
HATCH_ALPHA = 80
FILL_ALPHA = 22
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights/rtdetr_l_autoresearch_60ep.pt"


def load_font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def hatched_box(overlay, x1, y1, x2, y2, rgb, angle_up):
    w, h = max(int(x2 - x1), 2), max(int(y2 - y1), 2)
    tile = Image.new("RGBA", (w, h), rgb + (FILL_ALPHA,))
    td = ImageDraw.Draw(tile)
    for off in range(-h, w + h, HATCH_STEP):
        if angle_up:
            td.line([(off, h), (off + h, 0)], fill=rgb + (HATCH_ALPHA,), width=2)
        else:
            td.line([(off, 0), (off + h, h)], fill=rgb + (HATCH_ALPHA,), width=2)
    overlay.alpha_composite(tile, (int(x1), int(y1)))
    ImageDraw.Draw(overlay).rectangle([x1, y1, x2, y2],
                                      outline=rgb + (255,), width=OUTLINE_W)


def render(img_path: Path, result, out_path: Path):
    font, lfont = load_font(15), load_font(17)
    im = Image.open(img_path).convert("RGBA")
    b = result.boxes
    dets = [(float(b.conf[i]), int(b.cls[i]), b.xyxy[i].tolist())
            for i in range(len(b))]
    counts = {c: 0 for c in CLS}
    for _, c, _ in dets:
        counts[CLS[c]] += 1
    dets.sort(key=lambda d: (Z[d[1]], d[0]))       # class z, then confidence

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    for s, c, (x1, y1, x2, y2) in dets:
        hatched_box(overlay, x1, y1, x2, y2, COLOR[c], angle_up=(Z[c] < 100))
    im = Image.alpha_composite(im, overlay)

    d = ImageDraw.Draw(im)
    for s, c, (x1, y1, x2, y2) in dets:            # labels: non-wall only
        if c == 0:
            continue
        txt = f"{CLS[c]} {s:.2f}"
        tw = d.textlength(txt, font=font)
        ty = max(0, y1 - 19)
        d.rectangle([x1, ty, x1 + tw + 8, ty + 19], fill=COLOR[c] + (255,))
        d.text((x1 + 4, ty + 1), txt, fill="white", font=font)

    lx, ly = 8, 8                                   # legend with counts
    for c, cname in enumerate(CLS):
        label = f"{cname}: {counts[cname]}"
        tw = d.textlength(label, font=lfont)
        d.rectangle([lx - 4, ly - 3, lx + tw + 10, ly + 22],
                    fill=(255, 255, 255, 255), outline=COLOR[c] + (255,), width=3)
        d.text((lx + 3, ly), label, fill=COLOR[c] + (255,), font=lfont)
        lx += tw + 26

    im.convert("RGB").save(out_path)
    return counts, dets


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help="image file or folder of images")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold (canonical: 0.25)")
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="inference size; the model was trained at 1024 — "
                         "other sizes degrade accuracy")
    ap.add_argument("--device", default=None,
                    help="cuda:0 / cpu (default: auto)")
    ap.add_argument("--out-dir", default=None,
                    help="output folder (default: next to each input)")
    ap.add_argument("--save-json", action="store_true",
                    help="also write <name>_pred.json with raw boxes")
    args = ap.parse_args()

    src = Path(args.source)
    if src.is_dir():
        images = sorted(p for p in src.iterdir()
                        if p.suffix.lower() in IMG_EXTS
                        and not p.stem.endswith("_pred"))
    elif src.is_file():
        images = [src]
    else:
        sys.exit(f"source not found: {src}")
    if not images:
        sys.exit(f"no images found in {src}")

    import torch
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    from ultralytics import RTDETR
    model = RTDETR(args.weights)
    model.to(device)
    print(f"[predict] {Path(args.weights).name} on {device}, "
          f"conf>={args.conf}, imgsz={args.imgsz}, {len(images)} image(s)")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for p in images:
        r = model.predict(str(p), conf=args.conf, iou=0.5, imgsz=args.imgsz,
                          max_det=300, verbose=False, device=device)[0]
        out_base = (out_dir or p.parent) / (p.stem + "_pred")
        counts, dets = render(p, r, out_base.with_suffix(".png"))
        if args.save_json:
            with open(out_base.with_suffix(".json"), "w") as f:
                json.dump({"image": p.name, "conf_threshold": args.conf,
                           "classes": CLS,
                           "detections": [
                               {"cls": CLS[c], "conf": round(s, 4),
                                "xyxy": [round(v, 2) for v in xy]}
                               for s, c, xy in sorted(dets, key=lambda d: -d[0])]},
                          f, indent=1)
        print(f"  {p.name}: {sum(counts.values())} det {counts}"
              f" -> {out_base.name}.png")


if __name__ == "__main__":
    main()
