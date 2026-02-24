from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import List, Dict, Any

import pdfplumber
from llama_index.core import Document


@dataclass
class Line:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float


def _merge_bbox(lines: List[Line]) -> Dict[str, float]:
    return {
        "x0": min(l.x0 for l in lines),
        "top": min(l.top for l in lines),
        "x1": max(l.x1 for l in lines),
        "bottom": max(l.bottom for l in lines),
    }


def _looks_like_no_spaces(text: str) -> bool:
    t = text.strip()
    if len(t) < 25:
        return False
    return (" " not in t) and any(c.isalpha() for c in t)


def _extract_text_from_bbox(
    page: pdfplumber.page.Page,
    bbox: Dict[str, float],
) -> str | None:
    try:
        cropped = page.crop((bbox["x0"], bbox["top"], bbox["x1"], bbox["bottom"]))

        candidates: List[str] = []

        txt1 = cropped.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
        if txt1:
            candidates.append(" ".join(txt1.split()))

        txt2 = cropped.extract_text(x_tolerance=3, y_tolerance=3, layout=True)
        if txt2:
            candidates.append(" ".join(txt2.split()))

        txt3 = cropped.extract_text(x_tolerance=2, y_tolerance=2, layout=False)
        if txt3:
            candidates.append(" ".join(txt3.split()))

        for c in candidates:
            c = c.strip()
            if c and not _looks_like_no_spaces(c):
                return c

        for c in candidates:
            c = c.strip()
            if c:
                return c

        return None
    except Exception:
        return None


def load_pdf_as_paragraph_documents(
    pdf_path: Path,
    # paragraph break by indent increase (kept)
    max_indent_increase: float = 14.0,
    # adaptive gap settings
    gap_factor: float = 1.8,
    gap_min: float = 4.0,
    gap_epsilon: float = 0.5,
    # region filtering (MVP but very effective)
    header_frac: float = 0.08,
    footer_frac: float = 0.08,
    sidebar_frac: float = 0.22,
) -> List[Document]:
    """
    Layout-aware PDF loader with paragraph grouping + region filtering.

    Region filtering:
    - Drop header/footer bands by page height fractions
    - Drop left sidebar by page width fraction (common in academic PDFs)

    This dramatically reduces noise and prevents main-text line chopping.
    """
    documents: List[Document] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1

            page_width = float(page.width)
            page_height = float(page.height)

            header_y = page_height * header_frac
            footer_y = page_height * (1.0 - footer_frac)
            sidebar_x = page_width * sidebar_frac

            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            if not words:
                continue

            # Filter words by regions BEFORE grouping into lines
            filtered_words: List[Dict[str, Any]] = []
            for w in words:
                top = float(w["top"])
                bottom = float(w["bottom"])
                x0 = float(w["x0"])
                x1 = float(w["x1"])

                # header/footer bands
                if bottom <= header_y:
                    continue
                if top >= footer_y:
                    continue

                # left sidebar (only if the whole token is inside the sidebar)
                if x1 <= sidebar_x:
                    continue

                filtered_words.append(w)

            if not filtered_words:
                continue

            # Group word tokens by y-position (line candidates)
            raw_lines: Dict[float, List[Dict[str, Any]]] = {}
            for w in filtered_words:
                y_key = round(float(w["top"]), 1)
                raw_lines.setdefault(y_key, []).append(w)

            lines: List[Line] = []
            for _, ws in raw_lines.items():
                ws.sort(key=lambda x: float(x["x0"]))
                text = " ".join(str(x["text"]) for x in ws).strip()
                if not text:
                    continue

                x0 = min(float(x["x0"]) for x in ws)
                top = min(float(x["top"]) for x in ws)
                x1 = max(float(x["x1"]) for x in ws)
                bottom = max(float(x["bottom"]) for x in ws)

                line_bbox = {"x0": x0, "top": top, "x1": x1, "bottom": bottom}

                if _looks_like_no_spaces(text):
                    recovered = _extract_text_from_bbox(page, line_bbox)
                    if recovered:
                        text = recovered

                lines.append(Line(text=text, x0=x0, top=top, x1=x1, bottom=bottom))

            if not lines:
                continue

            lines.sort(key=lambda l: (l.top, l.x0))

            # Compute adaptive vertical gap threshold for this page
            gaps: List[float] = []
            for a, b in zip(lines, lines[1:]):
                g = b.top - a.bottom
                if g > 0:
                    gaps.append(g)

            med_gap = median(gaps) if gaps else gap_min
            gap_threshold = max(gap_min, med_gap * gap_factor) + gap_epsilon

            current: List[Line] = []
            para_index = 0

            def flush():
                nonlocal para_index, current
                if not current:
                    return
                para_index += 1
                bbox = _merge_bbox(current)
                paragraph_text = " ".join(l.text for l in current).strip()

                documents.append(
                    Document(
                        text=paragraph_text,
                        metadata={
                            "source_file": pdf_path.name,
                            "page": page_number,
                            "paragraph": para_index,
                            "bbox": bbox,
                        },
                    )
                )
                current = []

            prev: Line | None = None
            for line in lines:
                if prev is None:
                    current.append(line)
                    prev = line
                    continue

                vgap = line.top - prev.bottom
                indent_increase = (line.x0 - prev.x0) > max_indent_increase

                new_paragraph = (vgap > gap_threshold) or indent_increase

                if new_paragraph:
                    flush()

                current.append(line)
                prev = line

            flush()

    return documents
