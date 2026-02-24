# Architecture & Pipeline

## Goal
Build a local knowledge base for Q/A over mixed PDF + DOCX documents with citations to original sources (crop/page/pdf).

## Components
- PDF layout detection: `src/ingestion/layout_export_and_viz_pdf.py` (layoutparser + Detectron2 PubLayNet)
- OCR + canonicalization + export + chunking: `apps/rag-ba-ocr/*.py` (PaddleOCR)
- Indexing: `src/index/build_vector_index_from_chunks.py` (LlamaIndex + Chroma + multilingual-e5)
- Query: `src/index/test_query_with_citations.py` + manifest-based resolver
- CLI chat: `src/app/chat_cli.py` (RAG with history; simple MVP)

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

## Runtime
- Ollama must be running for Q/A and chat.
- Pipeline steps (layout/OCR/index) do not require Ollama.