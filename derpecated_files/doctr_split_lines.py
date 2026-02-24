from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import List, Tuple

import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# normalized box ((x0,y0),(x1,y1))
Box = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass
class Word:
    text: str
    box: Box


def denorm_box(box: Box, w: int, h: int) -> Tuple[int, int, int, int]:
    (x0, y0), (x1, y1) = box
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def merge_boxes(boxes: List[Box]) -> Box:
    x0 = min(b[0][0] for b in boxes)
    y0 = min(b[0][1] for b in boxes)
    x1 = max(b[1][0] for b in boxes)
    y1 = max(b[1][1] for b in boxes)
    return ((x0, y0), (x1, y1))


def split_line_by_adaptive_gaps(words: List[Word]) -> List[List[Word]]:
    """
    Adaptive split:
    - compute gaps between consecutive words by x
    - if the largest gap is an outlier (>> median gap) and is non-trivial,
      split at that gap (or multiple large gaps).

    This is robust when sidebar+main are on the same y-line.
    """
    if not words:
        return []

    words = sorted(words, key=lambda w: w.box[0][0])

    # compute gaps
    gaps = []
    for prev, cur in zip(words, words[1:]):
        gap = cur.box[0][0] - prev.box[1][0]
        gaps.append(gap)

    if not gaps:
        return [words]

    med = median([g for g in gaps if g > 0] or [0.0])
    mx = max(gaps)
    # Outlier test:
    # - absolute: at least 1.5% page width
    # - relative: at least 4x median (if median>0)
    abs_thr = 0.015
    rel_thr = (med * 4.0) if med > 0 else abs_thr

    split_thr = max(abs_thr, rel_thr)

    if mx < split_thr:
        return [words]

    # Split at all gaps that are close to the max gap (handles multi-column cases)
    # We take gaps >= 0.8 * mx to avoid over-splitting.
    cut_thr = 0.8 * mx

    segments: List[List[Word]] = [[words[0]]]
    for i, (prev, cur) in enumerate(zip(words, words[1:])):
        gap = gaps[i]
        if gap >= cut_thr:
            segments.append([cur])
        else:
            segments[-1].append(cur)

    return segments


def main() -> None:
    pdf_path = Path("data/raw/test.pdf")
    if not pdf_path.exists():
        print("Place test.pdf in data/raw/")
        return

    doc = DocumentFile.from_pdf(str(pdf_path))
    page_img = doc[0]  # RGB
    h, w = page_img.shape[:2]

    predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
    result = predictor(doc)
    page = result.pages[0]

    # Collect original line boxes to measure "wide lines"
    orig_lines = []
    for b in page.blocks:
        for ln in b.lines:
            orig_lines.append(ln.geometry)

    def width_norm(box: Box) -> float:
        return box[1][0] - box[0][0]

    wide_before = sum(1 for b in orig_lines if width_norm(b) >= 0.85)

    sublines: List[Tuple[Box, str]] = []
    split_examples = 0
    wide_after = 0

    for block in page.blocks:
        for ln in block.lines:
            ws: List[Word] = [Word(text=wd.value, box=wd.geometry) for wd in ln.words]

            segments = split_line_by_adaptive_gaps(ws)

            if len(segments) > 1 and split_examples < 8:
                split_examples += 1
                joined = [" ".join(w.text for w in seg) for seg in segments]
                print("\nSPLIT EXAMPLE:")
                for i, t in enumerate(joined, start=1):
                    print(f"  seg{i}: {t[:160]}")

            for seg in segments:
                box = merge_boxes([w.box for w in seg])
                text = " ".join(w.text for w in seg).strip()
                if width_norm(box) >= 0.85:
                    wide_after += 1
                sublines.append((box, text))

    print(f"\nPage1: original lines={len(orig_lines)}, wide_before(>=85% width)={wide_before}")
    print(f"Page1: sublines={len(sublines)}, wide_after(>=85% width)={wide_after}")

    img_bgr = cv2.cvtColor(page_img.copy(), cv2.COLOR_RGB2BGR)
    for box, _ in sublines:
        x0, y0, x1, y1 = denorm_box(box, w, h)
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)

    out = Path("doctr_sublines_page1.png")
    cv2.imwrite(str(out), img_bgr)
    print(f"Saved sub-line visualization to {out.resolve()}")


if __name__ == "__main__":
    main()
