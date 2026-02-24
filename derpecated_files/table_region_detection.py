from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np

BoxPx = Tuple[int, int, int, int]


def detect_table_regions(page_rgb: np.ndarray) -> List[BoxPx]:
    """
    Detect table regions using morphology-based line extraction.
    Much more stable than raw Hough on text baselines.
    """

    gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)

    # adaptive threshold
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5,
    )

    h, w = bw.shape

    # detect horizontal lines
    horizontal_kernel_len = max(25, w // 20)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_kernel_len, 1)
    )
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)

    # detect vertical lines
    vertical_kernel_len = max(25, h // 20)
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vertical_kernel_len)
    )
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)

    # combine
    table_mask = cv2.bitwise_or(horizontal, vertical)

    # find contours
    contours, _ = cv2.findContours(
        table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions: List[BoxPx] = []

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)

        # filter tiny regions
        if ww < w * 0.15 or hh < h * 0.10:
            continue

        regions.append((x, y, x + ww, y + hh))

    return regions
