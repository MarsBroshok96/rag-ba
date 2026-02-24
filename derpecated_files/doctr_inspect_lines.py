from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def denorm_box(box, w: int, h: int):
    # DocTR uses normalized coordinates in [0,1]
    (x0, y0), (x1, y1) = box
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def main() -> None:
    pdf_path = Path("data/raw/test.pdf")
    if not pdf_path.exists():
        print("Place test.pdf in data/raw/")
        return

    doc = DocumentFile.from_pdf(str(pdf_path))
    page_img = doc[0]  # numpy array RGB
    h, w = page_img.shape[:2]

    predictor = ocr_predictor(
        det_arch="db_resnet50",
        reco_arch="crnn_vgg16_bn",
        pretrained=True,
    )

    result = predictor(doc)
    page = result.pages[0]

    print(f"Page1: blocks={len(page.blocks)}")

    # Collect line boxes
    line_boxes = []
    for b in page.blocks:
        for ln in b.lines:
            line_boxes.append(ln.geometry)  # normalized box

    print(f"Page1: lines={len(line_boxes)}")

    # Print a few line boxes to see x-range (sidebar vs main)
    for i, box in enumerate(line_boxes[:15], start=1):
        x0, y0, x1, y1 = denorm_box(box, w, h)
        print(f"Line {i:02d}: x0={x0:4d} x1={x1:4d} y0={y0:4d} y1={y1:4d}")

    # Draw line boxes
    img = page_img.copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box in line_boxes:
        x0, y0, x1, y1 = denorm_box(box, w, h)
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (255, 0, 0), 2)

    out = Path("doctr_lines_page1.png")
    cv2.imwrite(str(out), img_bgr)
    print(f"\nSaved line-box visualization to {out.resolve()}")


if __name__ == "__main__":
    main()
