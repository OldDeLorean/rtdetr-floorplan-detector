"""
Polygon-based evaluation metrics for floor plan vectorization.

Extends standard COCO bbox mAP with:
- Polygon AP using actual polygon IoU (via Shapely)
- Per-class accuracy metrics
- Boundary IoU (BIoU) at multiple ratios
"""

import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.validation import make_valid


def polygon_iou(poly1, poly2):
    """Compute IoU between two polygons using Shapely.

    Args:
        poly1: (N, 2) array of polygon vertices.
        poly2: (M, 2) array of polygon vertices.

    Returns:
        float: IoU value in [0, 1].
    """
    try:
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)
        if not p1.is_valid:
            p1 = make_valid(p1)
        if not p2.is_valid:
            p2 = make_valid(p2)

        if p1.is_empty or p2.is_empty:
            return 0.0

        intersection = p1.intersection(p2).area
        union = p1.union(p2).area

        if union < 1e-6:
            return 0.0

        return intersection / union
    except Exception:
        return 0.0


def polygon_iou_matrix(pred_polys, gt_polys):
    """Compute IoU matrix between predicted and ground truth polygons.

    Args:
        pred_polys: List of (N, 2) arrays.
        gt_polys: List of (M, 2) arrays.

    Returns:
        (len(pred), len(gt)) IoU matrix.
    """
    n_pred = len(pred_polys)
    n_gt = len(gt_polys)
    iou_mat = np.zeros((n_pred, n_gt))

    for i in range(n_pred):
        for j in range(n_gt):
            iou_mat[i, j] = polygon_iou(pred_polys[i], gt_polys[j])

    return iou_mat


def boundary_iou(poly1, poly2, ratio=0.5):
    """Compute Boundary IoU between two polygons.

    BIoU focuses on the boundary region by dilating/eroding the
    intersection and union masks.

    Args:
        poly1: (N, 2) array of polygon vertices.
        poly2: (M, 2) array of polygon vertices.
        ratio: Boundary width as fraction of sqrt(area).

    Returns:
        float: Boundary IoU value.
    """
    try:
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)
        if not p1.is_valid:
            p1 = make_valid(p1)
        if not p2.is_valid:
            p2 = make_valid(p2)

        if p1.is_empty or p2.is_empty:
            return 0.0

        # Compute boundary width
        d1 = ratio * np.sqrt(p1.area)
        d2 = ratio * np.sqrt(p2.area)
        d = min(d1, d2)

        if d < 1e-6:
            return polygon_iou(poly1, poly2)

        # Create boundary regions
        boundary1 = p1.difference(p1.buffer(-d))
        boundary2 = p2.difference(p2.buffer(-d))

        if boundary1.is_empty or boundary2.is_empty:
            return 0.0

        intersection = boundary1.intersection(boundary2).area
        union = boundary1.union(boundary2).area

        if union < 1e-6:
            return 0.0

        return intersection / union
    except Exception:
        return 0.0


def compute_ap(recalls, precisions):
    """Compute Average Precision using 11-point interpolation.

    Args:
        recalls: Array of recall values.
        precisions: Array of precision values.

    Returns:
        float: Average Precision.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Monotonically decreasing precision
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = mrec >= t
        if mask.any():
            ap += mpre[mask].max() / 11.0

    return ap


def evaluate_polygon_ap(predictions, ground_truths, iou_thresholds=None,
                        use_polygon_iou=True):
    """Compute per-class polygon AP.

    Args:
        predictions: List of dicts per image, each with:
            'polygons': list of (N, 2) arrays
            'labels': list of int
            'scores': list of float
        ground_truths: List of dicts per image, each with:
            'polygons': list of (N, 2) arrays
            'labels': list of int
        iou_thresholds: List of IoU thresholds. Default: [0.5, 0.75].
        use_polygon_iou: If True, use polygon IoU. Otherwise, bbox IoU.

    Returns:
        Dict with per-class and overall AP at each threshold.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    # Collect all classes
    all_labels = set()
    for gt in ground_truths:
        all_labels.update(gt['labels'])

    results = {}

    for iou_thresh in iou_thresholds:
        class_aps = {}

        for cls in sorted(all_labels):
            # Gather predictions and GTs for this class across all images
            all_preds = []
            all_gts = []
            n_gts = 0

            for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                # Ground truth for this class
                gt_mask = [i for i, l in enumerate(gt['labels']) if l == cls]
                gt_polys = [gt['polygons'][i] for i in gt_mask]
                gt_matched = [False] * len(gt_polys)
                n_gts += len(gt_polys)
                all_gts.append((gt_polys, gt_matched))

                # Predictions for this class
                pred_mask = [i for i, l in enumerate(pred['labels']) if l == cls]
                for idx in pred_mask:
                    all_preds.append({
                        'polygon': pred['polygons'][idx],
                        'score': pred['scores'][idx],
                        'img_idx': img_idx,
                    })

            if n_gts == 0:
                class_aps[cls] = 0.0
                continue

            # Sort by score
            all_preds.sort(key=lambda p: p['score'], reverse=True)

            tp = np.zeros(len(all_preds))
            fp = np.zeros(len(all_preds))

            for pred_idx, pred_item in enumerate(all_preds):
                img_idx = pred_item['img_idx']
                gt_polys, gt_matched = all_gts[img_idx]

                if len(gt_polys) == 0:
                    fp[pred_idx] = 1
                    continue

                # Compute IoU with all GTs
                best_iou = 0.0
                best_gt = -1
                for gt_idx, gt_poly in enumerate(gt_polys):
                    if gt_matched[gt_idx]:
                        continue
                    if use_polygon_iou:
                        iou = polygon_iou(pred_item['polygon'], gt_poly)
                    else:
                        iou = _bbox_iou(pred_item['polygon'], gt_poly)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_idx

                if best_iou >= iou_thresh and best_gt >= 0:
                    tp[pred_idx] = 1
                    gt_matched[best_gt] = True
                else:
                    fp[pred_idx] = 1

            # Compute precision-recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recall = tp_cumsum / n_gts
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)

            class_aps[cls] = compute_ap(recall, precision)

        # Overall mAP
        if len(class_aps) > 0:
            mean_ap = np.mean(list(class_aps.values()))
        else:
            mean_ap = 0.0

        results[f'AP@{iou_thresh}'] = {
            'per_class': class_aps,
            'mAP': mean_ap,
        }

    return results


def evaluate_biou(predictions, ground_truths, ratios=None):
    """Compute Boundary IoU at multiple ratios.

    Args:
        predictions: Same format as evaluate_polygon_ap.
        ground_truths: Same format as evaluate_polygon_ap.
        ratios: List of boundary width ratios. Default: [0.1, 0.2, 0.5].

    Returns:
        Dict with mean BIoU at each ratio.
    """
    if ratios is None:
        ratios = [0.1, 0.2, 0.5]

    results = {}

    for ratio in ratios:
        all_bious = []

        for pred, gt in zip(predictions, ground_truths):
            for i in range(len(pred['polygons'])):
                best_biou = 0.0
                for j in range(len(gt['polygons'])):
                    if pred['labels'][i] == gt['labels'][j]:
                        biou = boundary_iou(
                            pred['polygons'][i], gt['polygons'][j], ratio=ratio
                        )
                        best_biou = max(best_biou, biou)
                if best_biou > 0:
                    all_bious.append(best_biou)

        results[f'BIoU@{ratio}'] = float(np.mean(all_bious)) if all_bious else 0.0

    return results


def _bbox_iou(poly1, poly2):
    """Compute IoU between bounding boxes of two polygons."""
    p1 = np.array(poly1)
    p2 = np.array(poly2)
    x1_1, y1_1 = p1.min(axis=0)
    x2_1, y2_1 = p1.max(axis=0)
    x1_2, y1_2 = p2.min(axis=0)
    x2_2, y2_2 = p2.max(axis=0)

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter

    if union < 1e-6:
        return 0.0
    return inter / union
