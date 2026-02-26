from __future__ import annotations

import json

from rag_ba.web.chat_store import ChatStore


def test_chat_store_create_append_update_delete(tmp_path):
    store = ChatStore(tmp_path / "chats")

    chat = store.create_chat()
    assert chat.title == "New chat"
    assert chat.settings.memory_k == 6
    assert (tmp_path / "chats" / f"{chat.chat_id}.json").exists()

    chat = store.append_message(
        chat.chat_id,
        role="user",
        content="Первое сообщение для генерации названия чата и проверки сохранения",
    )
    assert len(chat.messages) == 1
    assert chat.title != "New chat"
    assert "202" in chat.title

    chat = store.append_message(chat.chat_id, role="assistant", content="Ответ")
    assert [m.role for m in chat.messages] == ["user", "assistant"]

    chat = store.update_settings(chat.chat_id, memory_k=3)
    assert chat.settings.memory_k == 3

    summaries = store.list_chats()
    assert len(summaries) == 1
    assert summaries[0].chat_id == chat.chat_id
    assert summaries[0].message_count == 2

    payload = json.loads((tmp_path / "chats" / f"{chat.chat_id}.json").read_text(encoding="utf-8"))
    assert payload["settings"]["memory_k"] == 3
    assert payload["messages"][0]["role"] == "user"

    store.delete_chat(chat.chat_id)
    assert store.list_chats() == []


def test_chat_store_list_sorted_by_updated_at_desc(tmp_path):
    store = ChatStore(tmp_path / "chats")
    first = store.create_chat()
    second = store.create_chat()

    first = store.append_message(first.chat_id, role="user", content="old")
    second = store.append_message(second.chat_id, role="user", content="newer")

    # Make ordering deterministic even when local clock resolution is one second.
    first.updated_at = "2026-02-25T10:00:00+00:00"
    second.updated_at = "2026-02-25T10:00:01+00:00"
    store.save_chat(first)
    store.save_chat(second)

    summaries = store.list_chats()
    assert [s.chat_id for s in summaries] == [second.chat_id, first.chat_id]
