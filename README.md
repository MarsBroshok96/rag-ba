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
> при использовании иной версии, требуется поправить в src.index.test_query_with_citations.py и src.app.chat_cli.py
> в main строку: Settings.llm = Ollama(model="qwen2.5:14b-instruct", request_timeout=180.0)
- models/ (см. ниже)
- (Optional) NVIDIA GPU for faster OCR/layout

## Structure
- `src/` — приложение rag-ba (layout PDF, indexing, query, cli)
- `apps/rag-ba-ocr/` — отдельное приложение для OCR/каноникализации/экспорта/чанков (отдельный Poetry env)
- `data/inbox/`
  - `pdf/*.pdf`
  - `word/*.docx`
- `models/` — локальные артефакты Detectron2/PubLayNet (см. ниже)
- `data/manifest.json` — реестр документов и шаблоны путей для кликабельных источников (генерируется)
- `data/vectorstore/chroma_rag/` — persistent хранилище Chroma (генерируется)


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
cd ../..
```

## RAG Pipeline
1. Положить документы:
PDF: data/inbox/pdf/*.pdf
DOCX: data/inbox/word/*.docx

2. Запустить полный цикл формирования БД для RAG:
```bash
make all          # полный цикл
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

## Artefacts

layout/<doc_id>/... (генерируется)
apps/rag-ba-ocr/layout_ocr/<doc_id>/... (генерируется)
apps/rag-ba-ocr/canon/<doc_id>/... (генерируется)
apps/rag-ba-ocr/export/<doc_id>/{full_document.json,chunks.json} (генерируется)
data/vectorstore/chroma_rag/ (генерируется)

Эти папки обычно должны быть в .gitignore.