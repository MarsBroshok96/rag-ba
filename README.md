# rag-ba MVP (PDF+DOCX -> RAG with citations)

Локальная база знаний из смешанного inbox (PDF + DOCX) с:
- детекцией layout для PDF (PubLayNet / Detectron2),
- OCR по регионам (PaddleOCR, EN+RU),
- канонизацией блоков,
- чанкингом,
- индексацией в Chroma,
- Q/A с цитированием источников (ссылки на crop/page/pdf через manifest).

> В репозитории два Poetry-окружения: это сознательное решение из-за конфликтов зависимостей Detectron2 vs Paddle/PaddleOCR.


## Requirements
- Python 3.11
- Poetry
- git-lfs (для пула .pth моделей)
- Ollama + any local LLM (for Q/A, llm(RAG) chat)
> Сейчас qa mode и cli chat mode ожидают конкретную версию локальной LLM,
> при использовании иной версии, можно поменять `RAG_BA_OLLAMA_MODEL` (предпочтительно)
> или поправить `src/app/rag_chat_backend.py` / `src.index.test_query_with_citations.py`
- models/ (см. ниже)
- (Optional) NVIDIA GPU for faster OCR/layout

## Structure
- `src/` — приложение rag-ba (layout PDF, indexing, query, cli)
- `rag_ba/web/` — web chat UI (FastAPI + Bootstrap + disk-backed chat archive)
- `src/common/project_paths.py` — единая конфигурация repo-local путей (data/layout/apps/...); использовать в root-модулях вместо ручных `../..`
- `apps/rag-ba-ocr/` — отдельное приложение для OCR/каноникализации/экспорта/чанков (отдельный Poetry env)
- `apps/rag-ba-ocr/src/pipeline/` — реализации OCR/export/docx/chunks; файлы в `apps/rag-ba-ocr/*.py` оставлены как совместимые entrypoint-обёртки
- `apps/rag-ba-ocr/src/common/project_paths.py` — единая конфигурация путей для OCR-приложения (repo root, inbox, layout/export/canon/layout_ocr)
- `data/inbox/`
  - `pdf/*.pdf`
  - `word/*.docx`
- `models/` — локальные артефакты Detectron2/PubLayNet (см. ниже)
- `data/manifest.json` — реестр документов и шаблоны путей для кликабельных источников (генерируется)
- `data/vectorstore/chroma_rag/` — persistent хранилище Chroma (генерируется)
- `data/chats/` — локальное хранилище web-чатов (генерируется)


## Важно про `models/`

### Что лежит в `models/`
- `models/d2_configs/...` — YAML/py конфиги Detectron2 (локальная копия)
- `models/publaynet_frcnn_model_final.pth` (или `publaynet_frcnn_model_final.pth`) — веса PubLayNet
- (возможно) `models/publaynet.pth`
- `models/faster_rcnn_R_50_FPN_3x.yaml` (локальный ярлык/копия)

### Зачем это нужно
Скрипт `src/ingestion/layout_export_and_viz_pdf.py` использует **локальные пути**:
- config: `models/d2_configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`
- weights: `models/publaynet_frcnn_model_final.pth`

Поэтому для корректной работы `make layout` эти файлы должны присутствовать.

## Install
Root:
```bash
poetry install
cd apps/rag-ba-ocr
poetry install
poetry run pip install --no-cache-dir "paddlepaddle-gpu==3.3.0" \
  -f https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/
cd ../..
```

Optional local config (`.env`, loaded automatically by chat backend/web app):
```bash
cp .env.example .env
```

## RAG Pipeline
1. Положить документы:
PDF: data/inbox/pdf/*.pdf
DOCX: data/inbox/word/*.docx

2. Запустить полный цикл формирования БД для RAG:
```bash
make all          # полный цикл
make smoke        # быстрая проверка структуры/путей (без OCR/LLM)
make test         # pytest unit tests (после установки dev-зависимостей)
```
По шагам, если нужно дебажить:
```bash
make layout       # PDF → layout/<doc_id>/pageXXX_layout.json + png + viz
make ocr          # OCR по регионам → apps/rag-ba-ocr/layout_ocr/<doc_id>/
make canon        # OCR → canonical blocks → apps/rag-ba-ocr/canon/<doc_id>/
make export_pdf   # canonical → full_document.json (PDF) → apps/rag-ba-ocr/export/<doc_id>/
make docx         # DOCX → full_document.json → apps/rag-ba-ocr/export/<doc_id>/
make chunks       # full_document → chunks.json → apps/rag-ba-ocr/export/<doc_id>/
make index        # chunks → Chroma index → data/vectorstore/chroma_rag/
```
## Using LLM (with RAG):
1) Запустить Ollama

Для Q/A и chat Ollama должен быть запущен ollama:
```bash
ollama serve
```
Проверить, что модель доступна (пример):
```bash
ollama pull qwen2.5:14b-instruct
```
2) Запустить быстрый qa тест с одним вопросом (вопрос определяется константой QUESTION в src/index/test_query_with_citations.py)
```bash
make qa
```
3) Запустить интерактивный CLI чат (MVP) с памятью (объём определяется константой MAX_HISTORY в src/app/chat_cli.py)
```bash
make chat
```

4) Запустить Web чат (локально, dev, с авто-перезагрузкой)
```bash
make web
# или
poetry run uvicorn rag_ba.web.app:app --reload
# или
python -m rag_ba.web
```
Открыть: `http://127.0.0.1:8000`

Остановка:
- Используйте `Ctrl+C` (не `Ctrl+Z`).
- Если сервер был остановлен через `Ctrl+Z` и порт "залип":
```bash
make web_stop
make web
```
- Аналогично для Ollama:
```bash
make ollama_stop
make ollama_serve
```

### Web chat (MVP)
- Backend: FastAPI (`rag_ba/web/app.py`)
- Frontend: Bootstrap + vanilla JS (`rag_ba/web/templates/index.html`, `rag_ba/web/static/app.js`)
- Хранилище чатов: `data/chats/*.json` (переживает перезапуск сервера)
- Использует тот же RAG backend, что и CLI: `src/app/rag_chat_backend.py`
- Источники в UI кликабельны и открываются через локальный endpoint `/api/source/file?path=...`
- RAG retrieval для чата поддерживает двуязычный поиск (RU<->EN query expansion) и отвечает на языке вопроса
- Ответы LLM в чате используют 3 режима:
  - только по источникам,
  - по источникам + блок `ДОПОЛНЕНИЕ, не основанное на источниках:`,
  - `Информация в источниках отсутствует. ОТВЕТ от LLM:`

Зависимости Web UI добавлены в root `pyproject.toml`:
- `fastapi`
- `uvicorn`
- `jinja2`
- `python-dotenv` используется для автозагрузки `.env` (уже был в проекте)

## Artefacts

layout/<doc_id>/... (генерируется)
apps/rag-ba-ocr/layout_ocr/<doc_id>/... (генерируется)
apps/rag-ba-ocr/canon/<doc_id>/... (генерируется)
apps/rag-ba-ocr/export/<doc_id>/{full_document.json,chunks.json} (генерируется)
data/vectorstore/chroma_rag/ (генерируется)
data/chats/ (генерируется)

Эти папки обычно должны быть в .gitignore.

## Notes on project structure
- Индексация (`make index`) читает `chunks.json` из `apps/rag-ba-ocr/export/` **внутри этого репозитория**.
- Скрипты в `apps/rag-ba-ocr/*.py` — это entrypoint-обёртки для совместимости с `Makefile`; основная логика перенесена в `apps/rag-ba-ocr/src/pipeline/` и `apps/rag-ba-ocr/src/ingestion/`.
- Лёгкие unit-тесты вынесены в `tests/`; pytest настроен так, чтобы не коллектились CLI-скрипты вида `src/index/test_*.py`.
