# Agent Guide (Codex / Claude Code)

This repository is an MVP local RAG pipeline that builds a single knowledge base from mixed **PDF + DOCX** inbox.
The pipeline produces: layout → OCR → canonical → full_document → chunks → vector index (Chroma) → Q/A with citations.

## Ground rules for agents
- Do NOT commit generated artifacts (layout/, layout_ocr/, canon/, export/, vectorstore/). Keep `.gitignore` clean.
- Keep the pipeline deterministic: stable `doc_id`, stable chunk IDs, stable paths.
- Prefer small, localized patches; avoid refactors unless required.
- Changes must preserve backwards compatibility of JSON contracts unless explicitly bumped in docs.

## Monorepo layout
- `rag-ba/` (root): indexing, querying, CLI, PDF layout detection, orchestrating Makefile.
- `apps/rag-ba-ocr/`: OCR + canonicalization + export + chunking. Separate Poetry env.

## Pipeline overview (high-level)
1) PDF layout detection (rag-ba)
   - Input: `data/inbox/pdf/*.pdf`
   - Output per doc: `layout/<doc_id>/pageXXX.png`, `layout/<doc_id>/pageXXX_layout.json`, `layout/<doc_id>/pageXXX_layout_viz.png`

2) OCR over layout regions (apps/rag-ba-ocr)
   - Input: `../layout/<doc_id>/pageXXX_layout.json` + `pageXXX.png`
   - Output: `layout_ocr/<doc_id>/pageXXX_layout_ocr.json` + crops folder `pageXXX_crops/`

3) Canonical blocks (apps/rag-ba-ocr)
   - Input: `layout_ocr/<doc_id>/pageXXX_layout_ocr.json`
   - Output: `canon/<doc_id>/pageXXX_canonical.json`

4) Export full document (apps/rag-ba-ocr)
   - Input: `canon/<doc_id>/pageXXX_canonical.json`
   - Output: `export/<doc_id>/full_document.json` (+ optional xml/txt depending on script)

5) DOCX to full_document (apps/rag-ba-ocr)
   - Input: `data/inbox/word/*.docx`
   - Output: `export/<doc_id>/full_document.json` (single-page representation)

6) Chunk building (apps/rag-ba-ocr)
   - Input: `export/<doc_id>/full_document.json`
   - Output: `export/<doc_id>/chunks.json`

7) Vector index building (rag-ba)
   - Input: `apps/rag-ba-ocr/export/*/chunks.json`
   - Output: `data/vectorstore/chroma_rag/`

8) Query with citations (rag-ba)
   - Uses manifest for path resolution; prints sources with clickable PATH/URI.

## Make targets
From repo root:
- `make layout`
- `make ocr`
- `make canon`
- `make export_pdf`
- `make docx`
- `make chunks`
- `make index`
- `make qa`
- `make chat`
- `make all`

Important: `make ocr/canon/export/chunks/docx` run under the `apps/rag-ba-ocr` Poetry environment.

## Key design decisions
- Two Poetry envs are REQUIRED:
  - Detectron2/layoutparser stack conflicts with PaddleOCR stack on many setups.
- OCR is done per layout region and uses dual engines (en + ru) and selects best output.
- Metadata is aggressively slimmed before LlamaIndex node parsing to avoid "metadata length > chunk size" errors.
- `manifest.json` is the single source for resolving paths for citations (crop/page/pdf).

## When changing something
- If you change any JSON structure, update `DATA_CONTRACTS.md`.
- If you add a new artifact folder, update `.gitignore` and `ARCHITECTURE.md`.
- If you touch Makefile, verify `make all` pathing and env selection.

## Known fragile areas (do not break)
- detectron2 requires `torch` + `torchvision`; `layoutparser` imports `cv2` functions.
- PaddleOCR GPU requires correct PaddlePaddle build and CUDA libs; fallback to CPU is acceptable but slower.
- Chroma persistence path must stay in `data/vectorstore/chroma_rag/`.

## Quick smoke test for agents
1) `poetry install` in root
2) `cd apps/rag-ba-ocr && poetry install`
3) Put 1-2 PDFs into `data/inbox/pdf/` and 1 DOCX into `data/inbox/word/`
4) `make all`
5) `make qa`