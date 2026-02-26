from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse

from rag_ba.web.chat_store import ChatStore
from rag_ba.web.schemas import PatchChatSettingsRequest, SendChatRequest


@pytest.fixture()
def web_module_with_stubs(tmp_path, monkeypatch):
    import rag_ba.web.app as web_app_module

    class StubBackend:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.raise_error = False

        def generate_reply(self, *, user_text: str, messages: list[dict[str, str]], memory_k: int):
            self.calls.append(
                {
                    "user_text": user_text,
                    "messages": messages,
                    "memory_k": memory_k,
                }
            )
            if self.raise_error:
                raise RuntimeError("fake llm failure")

            filtered = [m for m in messages if m.get("role") in {"user", "assistant", "system"}]
            context_used = len(filtered[-memory_k:]) if memory_k > 0 else 0
            return SimpleNamespace(
                answer=f"stub-answer:{user_text}",
                sources_text="[1] score=1.0\n    PATH: /tmp/a.txt\n    URI:  file:///tmp/a.txt",
                context_messages_used=context_used,
            )

    stub_backend = StubBackend()
    monkeypatch.setattr(web_app_module, "chat_store", ChatStore(tmp_path / "chats"))
    monkeypatch.setattr(web_app_module, "rag_backend", stub_backend)
    return web_app_module, stub_backend


@pytest.mark.anyio
async def test_web_api_chat_crud_full_cycle(web_module_with_stubs):
    web, _ = web_module_with_stubs

    created = await web.create_chat()
    chat_id = created.chat_id
    assert created.title == "New chat"
    assert created.settings.memory_k == 6
    assert created.messages == []

    items = await web.list_chats()
    assert len(items) == 1
    assert items[0].chat_id == chat_id
    assert items[0].message_count == 0

    got = await web.get_chat(chat_id)
    assert got.chat_id == chat_id

    patched = await web.patch_chat_settings(chat_id, PatchChatSettingsRequest(memory_k=3))
    assert patched.settings.memory_k == 3

    deleted = await web.delete_chat(chat_id)
    assert deleted.ok is True

    items_after_delete = await web.list_chats()
    assert items_after_delete == []


@pytest.mark.anyio
async def test_web_api_send_uses_chat_settings_memory_k_and_persists_messages(web_module_with_stubs):
    web, stub_backend = web_module_with_stubs
    created = await web.create_chat()
    chat_id = created.chat_id

    await web.patch_chat_settings(chat_id, PatchChatSettingsRequest(memory_k=1))

    resp1 = await web.send_chat_message(SendChatRequest(chat_id=chat_id, message="Привет"))
    assert resp1.assistant_message.role == "assistant"
    assert resp1.assistant_message.content == "stub-answer:Привет"
    assert resp1.sources_text.startswith("[1]")
    assert resp1.context_messages_used == 1

    resp2 = await web.send_chat_message(SendChatRequest(chat_id=chat_id, message="Еще вопрос"))
    assert resp2.context_messages_used == 1

    assert len(stub_backend.calls) == 2
    assert stub_backend.calls[0]["memory_k"] == 1
    assert [m["role"] for m in stub_backend.calls[1]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]

    final_chat = await web.get_chat(chat_id)
    assert [m.role for m in final_chat.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.anyio
async def test_web_api_send_returns_json_error_and_keeps_server_alive(web_module_with_stubs):
    web, stub_backend = web_module_with_stubs
    created = await web.create_chat()
    chat_id = created.chat_id

    stub_backend.raise_error = True
    failed = await web.send_chat_message(SendChatRequest(chat_id=chat_id, message="test"))
    assert isinstance(failed, JSONResponse)
    assert failed.status_code == 500
    payload = json.loads(failed.body.decode("utf-8"))
    assert payload["error"] == "chat_send_failed"
    assert "fake llm failure" in payload["detail"]

    # User message is persisted before backend call; app should stay responsive.
    chat = await web.get_chat(chat_id)
    roles = [m.role for m in chat.messages]
    assert roles == ["user"]


@pytest.mark.anyio
async def test_web_api_not_found_errors_map_to_http_exceptions(web_module_with_stubs):
    web, _ = web_module_with_stubs

    with pytest.raises(HTTPException) as get_err:
        await web.get_chat("missing-id")
    assert get_err.value.status_code == 404

    with pytest.raises(HTTPException) as patch_err:
        await web.patch_chat_settings("missing-id", PatchChatSettingsRequest(memory_k=2))
    assert patch_err.value.status_code == 404

    with pytest.raises(HTTPException) as delete_err:
        await web.delete_chat("missing-id")
    assert delete_err.value.status_code == 404


@pytest.mark.anyio
async def test_web_api_source_file_endpoint_serves_repo_file(web_module_with_stubs):
    web, _ = web_module_with_stubs

    resp = await web.open_source_file(path="README.md")
    assert isinstance(resp, FileResponse)

    with pytest.raises(HTTPException) as outside_err:
        await web.open_source_file(path="/etc/hosts")
    assert outside_err.value.status_code in {403, 404}
