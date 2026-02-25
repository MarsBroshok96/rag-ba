.PHONY: help fmt lint typecheck \
        layout ocr canon export_pdf docx chunks index qa chat health \
        ollama_check ollama_serve smoke test \
        clean clean_layout clean_ocr clean_export clean_vector all

PY?=python
POETRY?=poetry

# --- Paths / modules ---
# rag-ba
LAYOUT_MOD=src.ingestion.layout_export_and_viz_pdf
INDEX_MOD=src.index.build_vector_index_from_chunks
QA_MOD=src.index.test_query_with_citations
HEALTH_MOD=src.index.health_check
CHAT_MOD=src.app.chat_cli
SMOKE_MOD=src.app.smoke_check

# rag-ba-ocr 
RAG_BA_OCR_DIR=apps/rag-ba-ocr
OCR_SCRIPT=ocr_regions.py
CANON_SCRIPT=build_canonical_doc.py
FULLDOC_SCRIPT=export_full_document.py
CHUNKS_SCRIPT=build_chunks.py
DOCX_SCRIPT=docx_to_full_documents.py

# generated artifacts
LAYOUT_DIR=layout
OCR_LAYOUT_DIR=$(RAG_BA_OCR_DIR)/layout_ocr
OCR_CANON_DIR=$(RAG_BA_OCR_DIR)/canon
OCR_EXPORT_DIR=$(RAG_BA_OCR_DIR)/export
VECTORSTORE_DIR=data/vectorstore/chroma_rag

help:
	@echo "Targets:"
	@echo "  make all            - layout -> ocr -> canon -> export_pdf + docx -> chunks -> index"
	@echo "  make layout         - run PDF layout detection for all PDFs in inbox"
	@echo "  make ocr            - run OCR over produced layouts (apps/rag-ba-ocr)"
	@echo "  make canon          - build canonical pages (apps/rag-ba-ocr)"
	@echo "  make export_pdf     - build full_document per PDF doc (apps/rag-ba-ocr)"
	@echo "  make docx           - build full_document per DOCX doc (apps/rag-ba-ocr)"
	@echo "  make chunks         - build chunks per doc (apps/rag-ba-ocr)"
	@echo "  make index          - build Chroma index from chunks"
	@echo "  make qa             - run local Q&A query_with_citations (requires Ollama)"
	@echo "  make chat           - start CLI chat with RAG + history (requires Ollama)"
	@echo "  make smoke          - quick repo/runtime readiness checks (no OCR/LLM execution)"
	@echo "  make test           - run pytest unit tests (lightweight core coverage)"
	@echo "  make ollama_check   - verify Ollama server is running"
	@echo "  make ollama_serve   - run 'ollama serve' (blocking)"
	@echo "  make clean          - remove generated artifacts"
	@echo "  make fmt/lint/typecheck/health"

fmt:
	$(POETRY) run ruff format .

lint:
	$(POETRY) run ruff check .

typecheck:
	$(POETRY) run mypy src

health:
	$(POETRY) run $(PY) -m $(HEALTH_MOD)

smoke:
	$(POETRY) run $(PY) -m $(SMOKE_MOD)

test:
	$(POETRY) run pytest -q

# ---- Ollama ----
OLLAMA_URL?=http://127.0.0.1:11434

ollama_check:
	@curl -fsS $(OLLAMA_URL)/api/tags >/dev/null 2>&1 || ( \
		echo "ERROR: Ollama server is not running."; \
		echo "Start it in another terminal:"; \
		echo "  ollama serve"; \
		echo "Then retry."; \
		exit 1 \
	)

# удобный таргет для запуска сервера вручную (блокирующий)
ollama_serve:
	@ollama serve

# ---- Pipeline steps ----

layout:
	$(POETRY) run $(PY) -m $(LAYOUT_MOD)

ocr:
	$(POETRY) -C $(RAG_BA_OCR_DIR) run $(PY) $(OCR_SCRIPT)

canon:
	$(POETRY) -C $(RAG_BA_OCR_DIR) run $(PY) $(CANON_SCRIPT)

export_pdf:
	$(POETRY) -C $(RAG_BA_OCR_DIR) run $(PY) $(FULLDOC_SCRIPT)

docx:
	$(POETRY) -C $(RAG_BA_OCR_DIR) run $(PY) $(DOCX_SCRIPT)

chunks:
	$(POETRY) -C $(RAG_BA_OCR_DIR) run $(PY) $(CHUNKS_SCRIPT)

index:
	$(POETRY) run $(PY) -m $(INDEX_MOD)

qa: ollama_check
	$(POETRY) run $(PY) -m $(QA_MOD)

all: layout ocr canon export_pdf docx chunks index
	@echo "DONE: all pipeline steps completed."

chat: ollama_check
	$(POETRY) run $(PY) -m $(CHAT_MOD)

# ---- Cleaning ----

clean_layout:
	rm -rf $(LAYOUT_DIR)/*

clean_ocr:
	rm -rf $(OCR_LAYOUT_DIR)/*
	rm -rf $(OCR_CANON_DIR)/*
	rm -rf $(OCR_EXPORT_DIR)/*

clean_export:
	rm -rf $(OCR_EXPORT_DIR)/*

clean_vector:
	rm -rf $(VECTORSTORE_DIR)/*

clean: clean_layout clean_ocr clean_vector
	@echo "Cleaned generated artifacts."
