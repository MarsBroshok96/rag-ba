from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Root app paths
DATA_DIR = PROJECT_ROOT / "data"
PDF_INBOX_DIR = DATA_DIR / "inbox" / "pdf"
WORD_INBOX_DIR = DATA_DIR / "inbox" / "word"
LAYOUT_ROOT = PROJECT_ROOT / "layout"
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = DATA_DIR / "manifest.json"
VECTORSTORE_ROOT = DATA_DIR / "vectorstore"
CHROMA_RAG_DIR = VECTORSTORE_ROOT / "chroma_rag"
CHROMA_DEMO_DIR = VECTORSTORE_ROOT / "chroma_demo"

# OCR app paths (second Poetry environment lives inside this repo)
RAG_BA_OCR_DIR = PROJECT_ROOT / "apps" / "rag-ba-ocr"
OCR_LAYOUT_ROOT = RAG_BA_OCR_DIR / "layout_ocr"
OCR_CANON_ROOT = RAG_BA_OCR_DIR / "canon"
OCR_EXPORT_ROOT = RAG_BA_OCR_DIR / "export"
