# Architecture & Pipeline

## Goal
Build a local knowledge base for Q/A over mixed PDF + DOCX documents with citations to original sources (crop/page/pdf).

## Components
- PDF layout detection: `src/ingestion/layout_export_and_viz_pdf.py` (layoutparser + Detectron2 PubLayNet)
- OCR + canonicalization + export + chunking:
  - compatibility entrypoints: `apps/rag-ba-ocr/*.py`
  - implementations: `apps/rag-ba-ocr/src/pipeline/*.py` and `apps/rag-ba-ocr/src/ingestion/*.py` (PaddleOCR)
- Indexing: `src/index/build_vector_index_from_chunks.py` (LlamaIndex + Chroma + multilingual-e5)
- Query: `src/index/test_query_with_citations.py` + manifest-based resolver
- CLI chat: `src/app/chat_cli.py` (RAG with history; simple MVP)
- Shared path config (root app): `src/common/project_paths.py` (single source of truth for repo-local dirs)

## Data flow
Inbox:
- `data/inbox/pdf/*.pdf`
- `data/inbox/word/*.docx`

Artifacts per doc_id:
- `layout/<doc_id>/`
  - `pageXXX.png`
  - `pageXXX_layout.json`
  - `pageXXX_layout_viz.png`

- `apps/rag-ba-ocr/layout_ocr/<doc_id>/`
  - `pageXXX_layout_ocr.json`
  - `pageXXX_crops/*.png`

- `apps/rag-ba-ocr/canon/<doc_id>/`
  - `pageXXX_canonical.json`

- `apps/rag-ba-ocr/export/<doc_id>/`
  - `full_document.json`
  - `chunks.json`

Index:
- `data/vectorstore/chroma_rag/` (persistent Chroma DB)

Manifest:
- `data/manifest.json` (doc registry + templates for paths)

## Path ownership (important)
- `rag-ba` (root app) and `apps/rag-ba-ocr` live in one repository and are expected to use **repo-local paths**.
- Root-side modules (`layout`, `index`, `qa`, `chat`) should resolve paths via `src/common/project_paths.py` to avoid drift.
- `src/index/build_vector_index_from_chunks.py` reads chunks from `apps/rag-ba-ocr/export/*/chunks.json` inside this repo (not from a sibling checkout).

## Entry points vs implementation
- `apps/rag-ba-ocr/build_canonical_doc.py` is a thin compatibility entrypoint.
- Canonicalization logic lives in `apps/rag-ba-ocr/src/ingestion/build_canonical_doc.py`.
- OCR scripts (`ocr_regions.py`, `export_full_document.py`, `docx_to_full_documents.py`, `build_chunks.py`) are also thin entrypoints; logic moved to `apps/rag-ba-ocr/src/pipeline/`.
- OCR-side path constants are centralized in `apps/rag-ba-ocr/src/common/project_paths.py`.
- `Makefile` keeps calling root script paths for backwards compatibility.

## Runtime
- Ollama must be running for Q/A and chat.
- Pipeline steps (layout/OCR/index) do not require Ollama.

## Verification
- `make smoke` runs lightweight structural checks (repo layout, key files, runtime directories, model/manifest presence as warn/ok).
- `make test` runs pytest unit tests for deterministic core helpers (paths/citation path resolution).
