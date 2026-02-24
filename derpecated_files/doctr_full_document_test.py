from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

Box = Tuple[Tuple[float, float], Tuple[float, float]]  # normalized
BoxPx = Tuple[int, int, int, int]  # pixels


@dataclass
class Word:
    text: str
    box: Box


@dataclass
class Segment:
    box: Box
    text: str


def denorm_box(box: Box, w: int, h: int) -> BoxPx:
    (x0, y0), (x1, y1) = box
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def merge_boxes(boxes: List[Box]) -> Box:
    x0 = min(b[0][0] for b in boxes)
    y0 = min(b[0][1] for b in boxes)
    x1 = max(b[1][0] for b in boxes)
    y1 = max(b[1][1] for b in boxes)
    return ((x0, y0), (x1, y1))


def width_norm(box: Box) -> float:
    return box[1][0] - box[0][0]


def iou_px(a: BoxPx, b: BoxPx) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / float(area_a + area_b - inter)


def union_px(a: BoxPx, b: BoxPx) -> BoxPx:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)


def merge_overlapping_boxes(boxes: List[BoxPx], iou_thr: float = 0.15) -> List[BoxPx]:
    merged: List[BoxPx] = []
    for box in sorted(boxes, key=lambda t: (t[1], t[0])):
        placed = False
        for i in range(len(merged)):
            if iou_px(merged[i], box) >= iou_thr:
                merged[i] = union_px(merged[i], box)
                placed = True
                break
        if not placed:
            merged.append(box)
    changed = True
    while changed:
        changed = False
        out: List[BoxPx] = []
        for box in merged:
            put = False
            for j in range(len(out)):
                if iou_px(out[j], box) >= iou_thr:
                    out[j] = union_px(out[j], box)
                    put = True
                    changed = True
                    break
            if not put:
                out.append(box)
        merged = out
    return merged


def split_line_by_adaptive_gaps(words: List[Word]) -> List[List[Word]]:
    if not words:
        return []
    words = sorted(words, key=lambda w: w.box[0][0])
    gaps = [cur.box[0][0] - prev.box[1][0] for prev, cur in zip(words, words[1:])]
    if not gaps:
        return [words]
    med = median([g for g in gaps if g > 0] or [0.0])
    mx = max(gaps)
    abs_thr = 0.015
    rel_thr = (med * 4.0) if med > 0 else abs_thr
    split_thr = max(abs_thr, rel_thr)
    if mx < split_thr:
        return [words]
    cut_thr = 0.8 * mx
    segments = [[words[0]]]
    for i, (_, cur) in enumerate(zip(words, words[1:])):
        if gaps[i] >= cut_thr:
            segments.append([cur])
        else:
            segments[-1].append(cur)
    return segments


def find_gutter_x_and_conf(page_rgb: np.ndarray) -> Tuple[int, float]:
    gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    top = int(h * 0.12)
    bottom = int(h * 0.88)
    roi = bw[top:bottom, :]
    col_sum = roi.sum(axis=0).astype(np.float32)
    k = max(31, (w // 80) | 1)
    col_smooth = cv2.GaussianBlur(col_sum.reshape(1, -1), (k, 1), 0).reshape(-1)
    left = int(w * 0.12)
    right = int(w * 0.88)
    search = col_smooth[left:right]
    x_rel = int(np.argmin(search))
    x = left + x_rel
    valley = float(search[x_rel])
    medv = float(np.median(search))
    conf = 0.0 if medv <= 1e-6 else max(0.0, min(1.0, 1.0 - (valley / medv)))
    return x, conf


def build_sublines(page_obj, page_rgb: np.ndarray, allow_gutter: bool) -> Dict[str, Any]:
    h, w = page_rgb.shape[:2]
    gutter_px, gutter_conf = find_gutter_x_and_conf(page_rgb)
    gutter_n = gutter_px / float(w)

    segments_words: List[List[Word]] = []
    for block in page_obj.blocks:
        for ln in block.lines:
            ws = [Word(text=wd.value, box=wd.geometry) for wd in ln.words]
            segments_words.extend(split_line_by_adaptive_gaps(ws))

    final_segments: List[List[Word]] = []
    for seg in segments_words:
        if not seg:
            continue
        seg = sorted(seg, key=lambda w_: w_.box[0][0])
        box = merge_boxes([w_.box for w_ in seg])
        crosses = box[0][0] < gutter_n < box[1][0]
        is_wide = width_norm(box) >= 0.85
        if allow_gutter and gutter_conf >= 0.35 and is_wide and crosses:
            left = []
            right = []
            for w_ in seg:
                cx = (w_.box[0][0] + w_.box[1][0]) / 2.0
                (left if cx < gutter_n else right).append(w_)
            if left:
                final_segments.append(sorted(left, key=lambda w_: w_.box[0][0]))
            if right:
                final_segments.append(sorted(right, key=lambda w_: w_.box[0][0]))
        else:
            final_segments.append(seg)

    sublines: List[Segment] = []
    for seg in final_segments:
        if not seg:
            continue
        box = merge_boxes([w_.box for w_ in seg])
        text = " ".join(w_.text for w_ in seg).strip()
        sublines.append(Segment(box=box, text=text))

    return {
        "gutter_px": gutter_px,
        "gutter_norm": round(gutter_n, 3),
        "gutter_conf": round(gutter_conf, 3),
        "sublines": sublines,
    }


def detect_ruled_region_from_rules(page_rgb: np.ndarray) -> List[BoxPx]:
    """
    Build table-like region from multiple long horizontal rules.
    Guarded against header/footer rules (otherwise page 1 becomes "table").
    """
    gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    h, w = bw.shape

    hk = max(60, w // 8)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)

    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rules: List[BoxPx] = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww < w * 0.60:
            continue
        if hh > max(6, int(h * 0.01)):
            continue
        # ignore header/footer band
        if y < int(h * 0.12) or y > int(h * 0.92):
            continue
        rules.append((x, y, x + ww, y + hh))

    if len(rules) < 2:
        return []

    x0 = min(r[0] for r in rules)
    x1 = max(r[2] for r in rules)
    y0 = min(r[1] for r in rules)
    y1 = max(r[3] for r in rules)

    # reject "rules spanning whole page" (header/footer)
    if (y1 - y0) > int(h * 0.55):
        return []

    pad_y = int(h * 0.03)
    pad_x = int(w * 0.02)
    reg = (max(0, x0 - pad_x), max(0, y0 - pad_y), min(w, x1 + pad_x), min(h, y1 + pad_y))
    return [reg]


def detect_table_regions_unruled(sublines: List[Segment], page_w: int, page_h: int) -> List[BoxPx]:
    if len(sublines) < 12:
        return []

    SIDEBAR_X0 = int(page_w * 0.18)

    items = []
    for seg in sublines:
        x0, y0, x1, y1 = denorm_box(seg.box, page_w, page_h)
        if x0 < SIDEBAR_X0:
            continue
        items.append((y0, x0, x1, y1, seg.text))
    items.sort(key=lambda t: (t[0], t[1]))

    WIN = 14
    MIN_ROWS = 8
    candidate_boxes: List[BoxPx] = []

    for start in range(0, len(items) - WIN + 1):
        window = items[start : start + WIN]
        texts = [t[4] for t in window]

        joined = " ".join(texts).lower()
        if "keyword" in joined or "key words" in joined:
            continue

        word_counts = [len(t.split()) for t in texts]
        short_ratio = sum(1 for c in word_counts if c <= 6) / float(WIN)
        if short_ratio < 0.60:
            continue

        x0s = sorted([t[1] for t in window])
        diffs = [b - a for a, b in zip(x0s, x0s[1:])]
        if not diffs:
            continue
        mx = max(diffs)
        if mx < page_w * 0.10:
            continue
        cut_idx = diffs.index(mx) + 1
        left = x0s[:cut_idx]
        right = x0s[cut_idx:]
        if len(left) < MIN_ROWS // 2 or len(right) < MIN_ROWS // 2:
            continue

        def spread(arr):
            return (np.percentile(arr, 80) - np.percentile(arr, 20)) if len(arr) >= 3 else 1e9

        if spread(left) > page_w * 0.06 or spread(right) > page_w * 0.06:
            continue

        x1s = [t[2] for t in window]
        if (np.percentile(x1s, 90) - np.percentile(x1s, 10)) < page_w * 0.10:
            continue

        xs0 = [t[1] for t in window]
        xs1 = [t[2] for t in window]
        ys0 = [t[0] for t in window]
        ys1 = [t[3] for t in window]

        x0 = max(0, min(xs0) - 6)
        x1 = min(page_w, max(xs1) + 6)
        y0 = max(0, min(ys0) - 6)
        y1 = min(page_h, max(ys1) + 6)

        candidate_boxes.append((x0, y0, x1, y1))

    return merge_overlapping_boxes(candidate_boxes, iou_thr=0.15)


def compute_text_coverage(reg: BoxPx, sublines: List[Segment], page_w: int, page_h: int) -> float:
    x0, y0, x1, y1 = reg
    area = max(1, (x1 - x0) * (y1 - y0))
    cov = 0
    for s in sublines:
        sx0, sy0, sx1, sy1 = denorm_box(s.box, page_w, page_h)
        ix0, iy0 = max(x0, sx0), max(y0, sy0)
        ix1, iy1 = min(x1, sx1), min(y1, sy1)
        if ix1 > ix0 and iy1 > iy0:
            cov += (ix1 - ix0) * (iy1 - iy0)
    return cov / float(area)


def graphics_density(reg: BoxPx, page_rgb: np.ndarray, sublines: List[Segment]) -> float:
    h, w = page_rgb.shape[:2]
    x0, y0, x1, y1 = reg
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    region = bw[y0:y1, x0:x1].copy()

    mask = np.zeros_like(region, dtype=np.uint8)
    for s in sublines:
        sx0, sy0, sx1, sy1 = denorm_box(s.box, w, h)
        if sx1 <= x0 or sx0 >= x1 or sy1 <= y0 or sy0 >= y1:
            continue
        rx0, ry0 = max(0, sx0 - x0), max(0, sy0 - y0)
        rx1, ry1 = min(x1 - x0, sx1 - x0), min(y1 - y0, sy1 - y0)
        cv2.rectangle(mask, (rx0, ry0), (rx1, ry1), 255, -1)

    region[mask > 0] = 0
    ink = int((region > 0).sum())
    area = int(region.size)
    return ink / float(max(1, area))


def subline_in_region_px(s: Segment, r: BoxPx, page_w: int, page_h: int) -> bool:
    sx0, sy0, sx1, sy1 = denorm_box(s.box, page_w, page_h)
    x0, y0, x1, y1 = r
    return sx0 >= x0 and sy0 >= y0 and sx1 <= x1 and sy1 <= y1


def tighten_region_to_members(region: BoxPx, members: List[Segment], page_w: int, page_h: int, pad_px: int = 8) -> Optional[BoxPx]:
    if not members:
        return None
    xs0, ys0, xs1, ys1 = [], [], [], []
    for s in members:
        x0, y0, x1, y1 = denorm_box(s.box, page_w, page_h)
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    x0 = max(0, min(xs0) - pad_px)
    y0 = max(0, min(ys0) - pad_px)
    x1 = min(page_w, max(xs1) + pad_px)
    y1 = min(page_h, max(ys1) + pad_px)
    return (x0, y0, x1, y1)


def regions_and_types(page_rgb: np.ndarray, sublines: List[Segment]) -> Tuple[List[BoxPx], List[BoxPx]]:
    h, w = page_rgb.shape[:2]
    ruled_reg = detect_ruled_region_from_rules(page_rgb)
    unruled = detect_table_regions_unruled(sublines, w, h)
    regs = merge_overlapping_boxes(ruled_reg + unruled, iou_thr=0.10)

    tables: List[BoxPx] = []
    figures: List[BoxPx] = []

    for r in regs:
        cov = compute_text_coverage(r, sublines, w, h)
        gd = graphics_density(r, page_rgb, sublines)

        if gd > 0.015:
            figures.append(r)
        elif cov >= 0.05:
            tables.append(r)

    tables = merge_overlapping_boxes(tables, 0.10)
    figures = merge_overlapping_boxes(figures, 0.10)

    # Tighten each region to its member sublines (removes extra paragraphs above)
    tight_tables: List[BoxPx] = []
    for r in tables:
        members = [s for s in sublines if subline_in_region_px(s, r, w, h)]
        t = tighten_region_to_members(r, members, w, h, pad_px=8)
        if t is not None:
            tight_tables.append(t)

    tight_figs: List[BoxPx] = []
    for r in figures:
        members = [s for s in sublines if subline_in_region_px(s, r, w, h)]
        t = tighten_region_to_members(r, members, w, h, pad_px=10)
        if t is not None:
            tight_figs.append(t)

    return merge_overlapping_boxes(tight_tables, 0.10), merge_overlapping_boxes(tight_figs, 0.10)


def subline_in_any_region(s: Segment, regions: List[BoxPx], page_w: int, page_h: int) -> bool:
    sx0, sy0, sx1, sy1 = denorm_box(s.box, page_w, page_h)
    for (x0, y0, x1, y1) in regions:
        if sx0 >= x0 and sy0 >= y0 and sx1 <= x1 and sy1 <= y1:
            return True
    return False


def decide_page_type(total_sublines: int, table_sublines: int, table_regions: List[BoxPx], page_h: int) -> str:
    if total_sublines == 0:
        return "image"

    ratio = table_sublines / float(total_sublines)
    y_cover = 0.0
    if table_regions:
        y0 = min(r[1] for r in table_regions)
        y1 = max(r[3] for r in table_regions)
        y_cover = (y1 - y0) / float(page_h)

    if (ratio >= 0.20 and y_cover >= 0.60) or y_cover >= 0.75:
        return "table"
    return "text"


def main():
    pdf_path = Path("data/raw/test.pdf")
    doc = DocumentFile.from_pdf(str(pdf_path))

    debug_dir = Path("debug_layout")
    debug_dir.mkdir(exist_ok=True)

    predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.det_predictor.model.to(device)
    predictor.reco_predictor.model.to(device)
    print(f"\nDocTR device: {device}\n")

    result = predictor(doc)
    print("Processing first 10 pages:\n")

    for i in range(min(10, len(result.pages))):
        page_idx = i + 1
        page_rgb = doc[i]
        page_obj = result.pages[i]
        h, w = page_rgb.shape[:2]

        seginfo = build_sublines(page_obj, page_rgb, allow_gutter=True)
        subs: List[Segment] = seginfo["sublines"]

        table_regs, fig_regs = regions_and_types(page_rgb, subs)
        excluded_regs = merge_overlapping_boxes(table_regs + fig_regs, 0.10)

        text_sublines = [s for s in subs if not subline_in_any_region(s, excluded_regs, w, h)]
        table_sublines = [s for s in subs if subline_in_any_region(s, table_regs, w, h)]
        fig_sublines = [s for s in subs if subline_in_any_region(s, fig_regs, w, h)]

        ptype = decide_page_type(len(subs), len(table_sublines), table_regs, h)

        if ptype == "table":
            seginfo = build_sublines(page_obj, page_rgb, allow_gutter=False)
            subs = seginfo["sublines"]
            table_regs, fig_regs = regions_and_types(page_rgb, subs)
            excluded_regs = merge_overlapping_boxes(table_regs + fig_regs, 0.10)
            text_sublines = [s for s in subs if not subline_in_any_region(s, excluded_regs, w, h)]
            table_sublines = [s for s in subs if subline_in_any_region(s, table_regs, w, h)]
            fig_sublines = [s for s in subs if subline_in_any_region(s, fig_regs, w, h)]

        img = cv2.cvtColor(page_rgb.copy(), cv2.COLOR_RGB2BGR)

        if ptype != "table":
            cv2.line(img, (seginfo["gutter_px"], 0), (seginfo["gutter_px"], h), (0, 255, 255), 2)

        for s in text_sublines:
            x0, y0, x1, y1 = denorm_box(s.box, w, h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

        for s in table_sublines:
            x0, y0, x1, y1 = denorm_box(s.box, w, h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)

        for s in fig_sublines:
            x0, y0, x1, y1 = denorm_box(s.box, w, h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 2)

        for (x0, y0, x1, y1) in table_regs:
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 4)

        for (x0, y0, x1, y1) in fig_regs:
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 165, 255), 4)

        out = debug_dir / f"page_{page_idx:02d}.png"
        cv2.imwrite(str(out), img)

        info = {
            "page": page_idx,
            "type": ptype,
            "text_sublines": len(text_sublines),
            "table_sublines": len(table_sublines),
            "figure_sublines": len(fig_sublines),
            "table_regions": len(table_regs),
            "figure_regions": len(fig_regs),
            "gutter_norm": seginfo["gutter_norm"],
            "gutter_conf": seginfo["gutter_conf"],
        }
        print(info)

    print("\nDebug images saved to:", debug_dir.resolve())


if __name__ == "__main__":
    main()
