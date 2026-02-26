.PHONY: help fmt lint typecheck \
        deps_layout deps_layout_fix \
        deps_ocr deps_ocr_fix \
        layout ocr canon export_pdf docx chunks index qa chat health \
        web web_stop \
        ollama_check ollama_serve ollama_stop smoke test \
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
WEB_APP=rag_ba.web.app:app
WEB_PORT?=8000

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

# detectron2 is intentionally bootstrapped via pip (no build isolation) because
# Poetry sync removes undeclared native packages and detectron2 often fails under
# build isolation without torch visible at build time.
DETECTRON2_GIT_SPEC?=git+https://github.com/facebookresearch/detectron2.git
OPENCV_PYTHON_VERSION?=4.13.0.92
PADDLE_GPU_VERSION?=3.3.0
PADDLE_GPU_WHEEL_URL?=https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/

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
	@echo "  make web            - start local web chat UI (FastAPI + Bootstrap)"
	@echo "  make web_stop       - stop local web chat UI process (uvicorn) if stuck after Ctrl+Z"
	@echo "  make smoke          - quick repo/runtime readiness checks (no OCR/LLM execution)"
	@echo "  make test           - run pytest unit tests (lightweight core coverage)"
	@echo "  make ollama_check   - verify Ollama server is running"
	@echo "  make ollama_serve   - run 'ollama serve' (blocking)"
	@echo "  make ollama_stop    - stop local 'ollama serve' if stuck after Ctrl+Z"
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
OLLAMA_PORT?=11434

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
	@$(MAKE) ollama_stop >/dev/null 2>&1 || true
	@sleep 1
	@ollama serve

ollama_stop:
	@command -v fuser >/dev/null 2>&1 && fuser -k $(OLLAMA_PORT)/tcp >/dev/null 2>&1 || true
	@command -v lsof >/dev/null 2>&1 && lsof -ti tcp:$(OLLAMA_PORT) | xargs -r kill >/dev/null 2>&1 || true
	@pkill -f '(^|/)ollama serve$$' >/dev/null 2>&1 || pkill -f 'ollama serve' >/dev/null 2>&1 || true
	@pkill -9 -f 'ollama serve' >/dev/null 2>&1 || true

# ---- Pipeline steps ----

deps_layout:
	@$(POETRY) run $(PY) -c "import torch, torchvision, torchvision.ops, detectron2, cv2; assert hasattr(cv2, 'getPerspectiveTransform')" >/dev/null 2>&1 || $(MAKE) deps_layout_fix

deps_layout_fix:
	@$(POETRY) run $(PY) -c "import torch, torchvision, torchvision.ops" >/dev/null 2>&1 || ( \
		echo "[deps_layout] ERROR: torch/torchvision missing or incompatible in root env."; \
		echo "[deps_layout] Install a matching pair first, then retry make."; \
		exit 1; \
	)
	@$(POETRY) run $(PY) -c "import cv2; assert hasattr(cv2, 'getPerspectiveTransform')" >/dev/null 2>&1 || ( \
		echo "[deps_layout] Reinstalling opencv-python ($(OPENCV_PYTHON_VERSION)) in root Poetry env..."; \
		$(POETRY) run pip install --force-reinstall --no-cache-dir "opencv-python==$(OPENCV_PYTHON_VERSION)"; \
	)
	@$(POETRY) run $(PY) -c "import detectron2" >/dev/null 2>&1 || ( \
		echo "[deps_layout] Reinstalling detectron2 in root Poetry env..."; \
		$(POETRY) run pip install -U --no-build-isolation "$(DETECTRON2_GIT_SPEC)"; \
	)
	@$(POETRY) run $(PY) -c "import torch, torchvision, torchvision.ops, detectron2, cv2; assert hasattr(cv2, 'getPerspectiveTransform')"

layout: deps_layout
	$(POETRY) run $(PY) -m $(LAYOUT_MOD)

deps_ocr:
	@$(POETRY) -C $(RAG_BA_OCR_DIR) run $(PY) -c "import cv2, paddle, paddleocr, paddlex; from paddleocr import PaddleOCR; assert hasattr(cv2, 'imdecode'); assert paddle.is_compiled_with_cuda()" >/dev/null 2>&1 || $(MAKE) deps_ocr_fix

deps_ocr_fix:
	@echo "[deps_ocr] Restoring declared OCR deps in $(RAG_BA_OCR_DIR) Poetry env..."
	@$(POETRY) -C $(RAG_BA_OCR_DIR) install
	@$(POETRY) -C $(RAG_BA_OCR_DIR) run $(PY) -c "import paddle; assert paddle.is_compiled_with_cuda()" >/dev/null 2>&1 || ( \
		echo "[deps_ocr] Installing paddlepaddle-gpu ($(PADDLE_GPU_VERSION))..."; \
		$(POETRY) -C $(RAG_BA_OCR_DIR) run pip uninstall -y paddlepaddle >/dev/null 2>&1 || true; \
		$(POETRY) -C $(RAG_BA_OCR_DIR) run pip install --no-cache-dir "paddlepaddle-gpu==$(PADDLE_GPU_VERSION)" -f "$(PADDLE_GPU_WHEEL_URL)"; \
	)
	@$(POETRY) -C $(RAG_BA_OCR_DIR) run $(PY) -c "import cv2, paddle, paddleocr, paddlex; from paddleocr import PaddleOCR; assert hasattr(cv2, 'imdecode'); assert paddle.is_compiled_with_cuda()"

ocr: deps_ocr
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

chat_probe: ollama_check
	@test -n "$(Q)" || (echo 'Usage: make chat_probe Q="your question"' && exit 2)
	$(POETRY) run $(PY) scripts/rag_chat_probe.py --question "$(Q)" --memory-k 6 --timeout-sec 240 --output /tmp/rag_chat_probe_result.json
	@echo "Result: /tmp/rag_chat_probe_result.json"

web:
	@$(MAKE) web_stop >/dev/null 2>&1 || true
	@sleep 1
	$(POETRY) run uvicorn $(WEB_APP) --reload

web_stop:
	@command -v fuser >/dev/null 2>&1 && fuser -k $(WEB_PORT)/tcp >/dev/null 2>&1 || true
	@command -v lsof >/dev/null 2>&1 && lsof -ti tcp:$(WEB_PORT) | xargs -r kill >/dev/null 2>&1 || true
	@pkill -f 'uvicorn .*rag_ba\.web\.app:app' >/dev/null 2>&1 || true
	@pkill -f 'rag_ba\.web\.app:app' >/dev/null 2>&1 || true
	@pkill -9 -f 'rag_ba\.web\.app:app' >/dev/null 2>&1 || true

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
