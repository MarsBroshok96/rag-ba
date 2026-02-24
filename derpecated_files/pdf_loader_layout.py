from __future__ import annotations

from pathlib import Path
from typing import List

import pdfplumber
from llama_index.core import Document


def load_pdf_with_layout(pdf_path: Path) -> List[Document]:
    """
    Layout-aware PDF loader using pdfplumber.
    Returns Documents with page and bounding box metadata.
    """

    documents: List[Document] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1

            # Extract text boxes instead of raw page text
            # Using page.extract_words to preserve layout grouping
            words = page.extract_words(
                use_text_flow=True,
                keep_blank_chars=False,
            )

            if not words:
                continue

            # Group words into simple line blocks based on y-position
            lines = {}
            for word in words:
                y_key = round(word["top"], 1)
                lines.setdefault(y_key, []).append(word)

            for line_words in lines.values():
                # Sort words left to right
                line_words.sort(key=lambda w: w["x0"])

                text = " ".join(w["text"] for w in line_words).strip()
                if not text:
                    continue

                # Bounding box of the line
                x0 = min(w["x0"] for w in line_words)
                top = min(w["top"] for w in line_words)
                x1 = max(w["x1"] for w in line_words)
                bottom = max(w["bottom"] for w in line_words)

                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source_file": pdf_path.name,
                            "page": page_number,
                            "bbox": {
                                "x0": x0,
                                "top": top,
                                "x1": x1,
                                "bottom": bottom,
                            },
                        },
                    )
                )

    return documents
