from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any

import pdfplumber
from llama_index.core import Document


BBox = Tuple[float, float, float, float]  # (x0, top, x1, bottom)


def _norm_ws(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip()


def _bbox_dict(b: BBox) -> Dict[str, float]:
    x0, top, x1, bottom = b
    return {"x0": x0, "top": top, "x1": x1, "bottom": bottom}


def _extract_region_text(page: pdfplumber.page.Page, region: BBox) -> str:
    cropped = page.crop(region)
    # layout=True often keeps word spacing better in PDFs
    txt = cropped.extract_text(layout=True, x_tolerance=2, y_tolerance=2)
    return _norm_ws(txt)


def _table_to_tsv(table: List[List[str | None]]) -> str:
    # Simple representation for MVP (keeps structure, RAG-friendly)
    lines = []
    for row in table:
        cells = [(_norm_ws(c) if c else "") for c in row]
        lines.append("\t".join(cells))
    return "\n".join(lines).strip()


def load_pdf_segmented(
    pdf_path: Path,
    header_frac: float = 0.08,
    footer_frac: float = 0.08,
    sidebar_frac: float = 0.22,
    include_sidebar: bool = True,
    include_tables: bool = True,
) -> List[Document]:
    """
    Segment page into:
      - main_text region (right area)
      - optional sidebar region (left area)
      - optional tables (as separate documents)

    This prevents sidebar/main mixing and keeps tables as whole entities.
    """
    docs: List[Document] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            w = float(page.width)
            h = float(page.height)

            header_y = h * header_frac
            footer_y = h * (1.0 - footer_frac)
            sidebar_x = w * sidebar_frac

            # Define regions (excluding header/footer by y-bounds)
            sidebar_region: BBox = (0.0, header_y, sidebar_x, footer_y)
            main_region: BBox = (sidebar_x, header_y, w, footer_y)

            # 1) Extract tables (whole) if requested
            if include_tables:
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []

                # pdfplumber extract_tables doesn't always give bbox; find_tables does.
                try:
                    found = page.find_tables()
                except Exception:
                    found = []

                # If we have bbox info, attach it; otherwise keep bbox=None.
                for ti, t in enumerate(tables, start=1):
                    tsv = _table_to_tsv(t)
                    if not tsv:
                        continue

                    bbox = None
                    if ti - 1 < len(found):
                        try:
                            bbox = found[ti - 1].bbox  # type: ignore[attr-defined]
                        except Exception:
                            bbox = None

                    meta: Dict[str, Any] = {
                        "source_file": pdf_path.name,
                        "page": page_number,
                        "region_type": "table",
                        "table_index": ti,
                    }
                    if bbox:
                        meta["bbox"] = _bbox_dict((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))

                    docs.append(Document(text=tsv, metadata=meta))

            # 2) Extract main text region as ONE document per page (MVP)
            main_text = _extract_region_text(page, main_region)
            if main_text:
                docs.append(
                    Document(
                        text=main_text,
                        metadata={
                            "source_file": pdf_path.name,
                            "page": page_number,
                            "region_type": "main_text",
                            "bbox": _bbox_dict(main_region),
                        },
                    )
                )

            # 3) Extract sidebar region separately (optional)
            if include_sidebar:
                side_text = _extract_region_text(page, sidebar_region)
                if side_text:
                    docs.append(
                        Document(
                            text=side_text,
                            metadata={
                                "source_file": pdf_path.name,
                                "page": page_number,
                                "region_type": "sidebar",
                                "bbox": _bbox_dict(sidebar_region),
                            },
                        )
                    )

    return docs
