from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_MEMORY_K = 6
DEFAULT_CHAT_TITLE = "New chat"


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


class ChatSettings(BaseModel):
    memory_k: int = Field(default=DEFAULT_MEMORY_K, ge=0, le=100)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)
    ts: str


class ChatFile(BaseModel):
    chat_id: str
    title: str = DEFAULT_CHAT_TITLE
    created_at: str
    updated_at: str
    messages: list[ChatMessage] = Field(default_factory=list)
    settings: ChatSettings = Field(default_factory=ChatSettings)


class ChatSummary(BaseModel):
    chat_id: str
    title: str
    updated_at: str
    created_at: str
    message_count: int = 0


class CreateChatRequest(BaseModel):
    settings: ChatSettings | None = None


class SendChatRequest(BaseModel):
    chat_id: str
    message: str = Field(min_length=1, max_length=10000)


class PatchChatSettingsRequest(BaseModel):
    memory_k: int | None = Field(default=None, ge=0, le=100)


class ChatSendResponse(BaseModel):
    chat: ChatFile
    assistant_message: ChatMessage
    sources_text: str
    context_messages_used: int


class OkResponse(BaseModel):
    ok: bool = True

