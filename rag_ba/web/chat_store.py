from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from rag_ba.web.schemas import (
    DEFAULT_CHAT_TITLE,
    ChatFile,
    ChatMessage,
    ChatSettings,
    ChatSummary,
    utc_now_iso,
)


class ChatStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _chat_path(self, chat_id: str) -> Path:
        if not chat_id or "/" in chat_id or "\\" in chat_id:
            raise ValueError("invalid chat_id")
        return self.base_dir / f"{chat_id}.json"

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            temp_path = Path(tmp.name)
        temp_path.replace(path)

    @staticmethod
    def _make_title_from_first_user_message(content: str, ts: str) -> str:
        text = " ".join(content.split())
        words = text.split(" ")
        short = " ".join(words[:8]).strip()
        if len(words) > 8:
            short += "..."
        date_part = ts[:10]
        return f"{short or DEFAULT_CHAT_TITLE} ({date_part})"

    @staticmethod
    def _to_summary(chat: ChatFile) -> ChatSummary:
        return ChatSummary(
            chat_id=chat.chat_id,
            title=chat.title,
            updated_at=chat.updated_at,
            created_at=chat.created_at,
            message_count=len(chat.messages),
        )

    def create_chat(self, *, settings: ChatSettings | None = None) -> ChatFile:
        now = utc_now_iso()
        chat = ChatFile(
            chat_id=str(uuid4()),
            title=DEFAULT_CHAT_TITLE,
            created_at=now,
            updated_at=now,
            messages=[],
            settings=settings or ChatSettings(),
        )
        self.save_chat(chat)
        return chat

    def save_chat(self, chat: ChatFile) -> ChatFile:
        self._write_json_atomic(self._chat_path(chat.chat_id), chat.model_dump(mode="json"))
        return chat

    def get_chat(self, chat_id: str) -> ChatFile:
        path = self._chat_path(chat_id)
        if not path.exists():
            raise FileNotFoundError(chat_id)
        return ChatFile.model_validate_json(path.read_text(encoding="utf-8"))

    def list_chats(self) -> list[ChatSummary]:
        chats: list[ChatSummary] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                chat = ChatFile.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            chats.append(self._to_summary(chat))
        chats.sort(key=lambda c: c.updated_at, reverse=True)
        return chats

    def delete_chat(self, chat_id: str) -> None:
        path = self._chat_path(chat_id)
        if not path.exists():
            raise FileNotFoundError(chat_id)
        path.unlink()

    def update_settings(self, chat_id: str, *, memory_k: int | None = None) -> ChatFile:
        chat = self.get_chat(chat_id)
        if memory_k is not None:
            chat.settings = ChatSettings(memory_k=memory_k)
            chat.updated_at = utc_now_iso()
        self.save_chat(chat)
        return chat

    def append_message(self, chat_id: str, *, role: str, content: str) -> ChatFile:
        chat = self.get_chat(chat_id)
        ts = utc_now_iso()
        msg = ChatMessage(role=role, content=content, ts=ts)
        chat.messages.append(msg)
        if role == "user" and chat.title == DEFAULT_CHAT_TITLE and len(
            [m for m in chat.messages if m.role == "user"]
        ) == 1:
            chat.title = self._make_title_from_first_user_message(content, ts)
        chat.updated_at = ts
        self.save_chat(chat)
        return chat

