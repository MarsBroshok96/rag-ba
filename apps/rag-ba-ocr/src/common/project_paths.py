from __future__ import annotations

from pathlib import Path


OCR_APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]

OCR_LAYOUT_ROOT = OCR_APP_ROOT / "layout_ocr"
OCR_CANON_ROOT = OCR_APP_ROOT / "canon"
OCR_EXPORT_ROOT = OCR_APP_ROOT / "export"

ROOT_LAYOUT_ROOT = REPO_ROOT / "layout"
WORD_INBOX_DIR = REPO_ROOT / "data" / "inbox" / "word"

