from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader
from llama_index.core import Document


def split_into_paragraphs(text: str) -> List[str]:
    """
    Basic paragraph splitter.
    Later we can replace with smarter logic (layout-aware / OCR-aware).
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs


def load_pdf_as_documents(pdf_path: Path) -> List[Document]:
    """
    Load PDF and return list of LlamaIndex Documents
    with page and paragraph metadata preserved.
    """

    reader = PdfReader(str(pdf_path))
    documents: List[Document] = []

    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1
        text = page.extract_text()

        if not text:
            continue

        paragraphs = split_into_paragraphs(text)

        for paragraph_index, paragraph in enumerate(paragraphs):
            documents.append(
                Document(
                    text=paragraph,
                    metadata={
                        "source_file": pdf_path.name,
                        "page": page_number,
                        "paragraph": paragraph_index + 1,
                    },
                )
            )

    return documents
