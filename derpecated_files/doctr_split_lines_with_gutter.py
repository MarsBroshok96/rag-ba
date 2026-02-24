from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import List, Tuple

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

Box = Tuple[Tuple[float, float], Tuple[float, float]]  # normalized


@dataclass
class Word:
    text: str
    box: Box


def denorm_box(box: Box, w: int, h: int) -> Tuple[int, int, int, int]:
    (x0, y0), (x1, y1) = box
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def norm_x(px: int, w: int) -> float:
    return max(0.0, min(1.0, px / float(w)))


def merge_boxes(boxes: List[Box]) -> Box:
    x0 = min(b[0][0] for b in boxes)
    y0 = min(b[0][1] for b in boxes)
    x1 = max(b[1][0] for b in boxes)
    y1 = max(b[1][1] for b in boxes)
    return ((x0, y0), (x1, y1))


def width_norm(box: Box) -> float:
    return box[1][0] - box[0][0]


def split_line_by_adaptive_gaps(words: List[Word]) -> List[List[Word]]:
    if not words:
        return []

    words = sorted(words, key=lambda w: w.box[0][0])

    gaps = []
    for prev, cur in zip(words, words[1:]):
        gaps.append(cur.box[0][0] - prev.box[1][0])

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
    segments: List[List[Word]] = [[words[0]]]
    for i, (_, cur) in enumerate(zip(words, words[1:])):
        if gaps[i] >= cut_thr:
            segments.append([cur])
        else:
            segments[-1].append(cur)
    return segments


def find_gutter_x(page_rgb: np.ndarray) -> int:
    """
    Find a strong vertical whitespace separator (gutter) on the page image.
    Returns x pixel coordinate of the best cut.
    """
    gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
    # binarize: text ~ black -> 1, background -> 0
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = bw.shape

    # Remove very top/bottom bands to ignore header/footer noise
    top = int(h * 0.12)
    bottom = int(h * 0.88)
    roi = bw[top:bottom, :]

    # vertical projection: amount of ink per column
    col_sum = roi.sum(axis=0).astype(np.float32)

    # smooth to ignore small gaps between words
    k = max(31, (w // 80) | 1)  # odd kernel size
    col_smooth = cv2.GaussianBlur(col_sum.reshape(1, -1), (k, 1), 0).reshape(-1)

    # We want a LOW value (whitespace). Avoid edges (margins).
    left = int(w * 0.12)
    right = int(w * 0.88)

    search = col_smooth[left:right]
    x_rel = int(np.argmin(search))
    x = left + x_rel
    return x


def split_box_by_gutter(box: Box, text: str, gutter_x_norm: float) -> List[Tuple[Box, str]]:
    """
    Split a wide box by gutter_x if it crosses the gutter.
    We split the TEXT approximately by assigning words is not available here,
    so for MVP we just keep same text in both segments? No.
    We'll only split geometrically and keep text empty for the cut parts is bad.
    So we will use this splitter ONLY to split WORD-based segments (below).
    """
    raise RuntimeError("Not used directly")


def main() -> None:
    pdf_path = Path("data/raw/test.pdf")
    if not pdf_path.exists():
        print("Place test.pdf in data/raw/")
        return

    doc = DocumentFile.from_pdf(str(pdf_path))
    page_rgb = doc[0]
    h, w = page_rgb.shape[:2]

    gutter_x = find_gutter_x(page_rgb)
    gutter_x_n = norm_x(gutter_x, w)
    print(f"Gutter x (px)={gutter_x}, norm={gutter_x_n:.3f}")

    predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
    result = predictor(doc)
    page = result.pages[0]

    orig_lines = [ln.geometry for b in page.blocks for ln in b.lines]
    wide_before = sum(1 for b in orig_lines if width_norm(b) >= 0.85)

    # First: adaptive split by word gaps
    segments_words: List[List[Word]] = []
    for b in page.blocks:
        for ln in b.lines:
            ws: List[Word] = [Word(text=wd.value, box=wd.geometry) for wd in ln.words]
            segments_words.extend(split_line_by_adaptive_gaps(ws))

    # Second: if a segment box is still wide AND crosses gutter, split words by gutter
    final_segments: List[List[Word]] = []
    for seg in segments_words:
        if not seg:
            continue
        seg = sorted(seg, key=lambda w_: w_.box[0][0])
        box = merge_boxes([w_.box for w_ in seg])

        crosses_gutter = box[0][0] < gutter_x_n < box[1][0]
        if width_norm(box) >= 0.85 and crosses_gutter:
            left_words = [w_ for w_ in seg if w_.box[1][0] <= gutter_x_n]
            right_words = [w_ for w_ in seg if w_.box[0][0] >= gutter_x_n]
            mid_words = [w_ for w_ in seg if w_ not in left_words and w_ not in right_words]

            # Assign mid words by center
            for w_ in mid_words:
                cx = (w_.box[0][0] + w_.box[1][0]) / 2.0
                if cx < gutter_x_n:
                    left_words.append(w_)
                else:
                    right_words.append(w_)

            if left_words:
                final_segments.append(sorted(left_words, key=lambda w_: w_.box[0][0]))
            if right_words:
                final_segments.append(sorted(right_words, key=lambda w_: w_.box[0][0]))
            if not left_words and not right_words:
                final_segments.append(seg)
        else:
            final_segments.append(seg)

    # Build final sublines
    sublines: List[Tuple[Box, str]] = []
    wide_after1 = 0
    wide_after2 = 0

    for seg in segments_words:
        if not seg:
            continue
        b = merge_boxes([w_.box for w_ in seg])
        if width_norm(b) >= 0.85:
            wide_after1 += 1

    for seg in final_segments:
        if not seg:
            continue
        b = merge_boxes([w_.box for w_ in seg])
        t = " ".join(w_.text for w_ in seg).strip()
        if width_norm(b) >= 0.85:
            wide_after2 += 1
        sublines.append((b, t))

    print(f"Page1: original lines={len(orig_lines)}, wide_before(>=85% width)={wide_before}")
    print(f"Page1: sublines_after_gap_split={len(segments_words)}, wide_after1={wide_after1}")
    print(f"Page1: final_sublines={len(sublines)}, wide_after2={wide_after2}")

    # Draw boxes + gutter line
    img = cv2.cvtColor(page_rgb.copy(), cv2.COLOR_RGB2BGR)
    cv2.line(img, (gutter_x, 0), (gutter_x, h), (0, 255, 255), 2)  # yellow gutter

    for box, _ in sublines:
        x0, y0, x1, y1 = denorm_box(box, w, h)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

    out = Path("doctr_sublines_gutter_page1.png")
    cv2.imwrite(str(out), img)
    print(f"Saved visualization to {out.resolve()}")


if __name__ == "__main__":
    main()
