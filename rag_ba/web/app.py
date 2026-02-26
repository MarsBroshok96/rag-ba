from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.params import Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_ba.web.chat_store import ChatStore
from rag_ba.web.schemas import (
    ChatFile,
    ChatSendResponse,
    ChatSummary,
    CreateChatRequest,
    OkResponse,
    PatchChatSettingsRequest,
    SendChatRequest,
)
from src.common.project_paths import DATA_DIR, PROJECT_ROOT

load_dotenv()

WEB_DIR = Path(__file__).resolve().parent
CHAT_DATA_DIR = DATA_DIR / "chats"

app = FastAPI(title="rag-ba Web Chat", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

chat_store = ChatStore(CHAT_DATA_DIR)
rag_backend: Any | None = None


def get_rag_backend() -> Any:
    global rag_backend
    if rag_backend is None:
        from src.app.rag_chat_backend import RagChatBackend

        rag_backend = RagChatBackend()
    return rag_backend


def _resolve_project_file(path_value: str) -> Path:
    if not path_value:
        raise HTTPException(status_code=400, detail="path is required")
    try:
        raw = Path(path_value)
        candidate = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="file not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid path: {path_value}") from exc

    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path outside project root is not allowed") from exc

    if not resolved.is_file():
        raise HTTPException(status_code=400, detail="path is not a file")
    return resolved


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "app_title": "rag-ba Web Chat"},
    )


@app.get("/api/source/file")
async def open_source_file(path: str = Query(..., min_length=1)) -> FileResponse:
    resolved = _resolve_project_file(path)
    return FileResponse(
        path=str(resolved),
        filename=resolved.name,
        content_disposition_type="inline",
    )


@app.get("/api/chats", response_model=list[ChatSummary])
async def list_chats() -> list[ChatSummary]:
    return chat_store.list_chats()


@app.post("/api/chats", response_model=ChatFile)
async def create_chat(payload: CreateChatRequest | None = None) -> ChatFile:
    settings = payload.settings if payload else None
    return chat_store.create_chat(settings=settings)


@app.get("/api/chats/{chat_id}", response_model=ChatFile)
async def get_chat(chat_id: str) -> ChatFile:
    try:
        return chat_store.get_chat(chat_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/chats/{chat_id}", response_model=OkResponse)
async def delete_chat(chat_id: str) -> OkResponse:
    try:
        chat_store.delete_chat(chat_id)
        return OkResponse(ok=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/chats/{chat_id}/settings", response_model=ChatFile)
async def patch_chat_settings(chat_id: str, payload: PatchChatSettingsRequest) -> ChatFile:
    try:
        return chat_store.update_settings(chat_id, memory_k=payload.memory_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat/send", response_model=ChatSendResponse)
async def send_chat_message(payload: SendChatRequest) -> ChatSendResponse | JSONResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is empty")

    try:
        chat = chat_store.append_message(payload.chat_id, role="user", content=message)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"chat not found: {payload.chat_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        backend = rag_backend if rag_backend is not None else get_rag_backend()
        reply = backend.generate_reply(
            user_text=message,
            messages=[m.model_dump(mode="json") for m in chat.messages],
            memory_k=chat.settings.memory_k,
        )
        chat = chat_store.append_message(
            payload.chat_id,
            role="assistant",
            content=reply.answer,
            sources_text=reply.sources_text,
        )
        assistant_message = chat.messages[-1]
        return ChatSendResponse(
            chat=chat,
            assistant_message=assistant_message,
            sources_text=reply.sources_text,
            context_messages_used=reply.context_messages_used,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "chat_send_failed",
                "detail": str(exc),
            },
        )


def main() -> None:
    import uvicorn

    host = os.getenv("RAG_BA_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("RAG_BA_WEB_PORT", "8000"))
    reload = os.getenv("RAG_BA_WEB_RELOAD", "1") not in {"0", "false", "False"}
    uvicorn.run("rag_ba.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
