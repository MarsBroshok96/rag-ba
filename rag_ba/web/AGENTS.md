# rag_ba/web Agent Notes

## Purpose
Local FastAPI + Bootstrap web chat UI for the existing `rag-ba` RAG/LLM backend.

## Rules for changes
- Reuse `src.app.rag_chat_backend` for RAG/LLM calls; do not duplicate retrieval/prompt logic.
- Keep chat history persistence local in `data/chats/*.json` (one file per chat).
- Preserve `chat_cli` behavior and compatibility.
- Keep API contracts backward-compatible for the Web MVP unless `DATA_CONTRACTS.md` is updated.

## Key files
- `rag_ba/web/app.py` — FastAPI routes + UI entry
- `rag_ba/web/chat_store.py` — disk persistence CRUD
- `rag_ba/web/schemas.py` — Pydantic models for file/API contracts
- `rag_ba/web/templates/index.html` — UI markup
- `rag_ba/web/static/app.js` — frontend logic
- `src/app/rag_chat_backend.py` — shared RAG backend (CLI + Web), includes retrieval and prompt rules

## Dev run
- `make web`
- `make web_stop` (if port remains occupied after `Ctrl+Z`)
- or `poetry run uvicorn rag_ba.web.app:app --reload`
- or `python -m rag_ba.web`

## Storage
- Chats: `data/chats/<chat_id>.json`
