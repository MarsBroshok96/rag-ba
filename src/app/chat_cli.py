from __future__ import annotations

import os

from src.app.rag_chat_backend import RagChatBackend

MAX_HISTORY = 6


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    backend = RagChatBackend()

    history: list[dict[str, str]] = []

    print("RAG Chat CLI")
    print("Type /exit to quit, /clear to reset history\n")

    while True:
        user_input = input(">>> ").strip()

        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input == "/clear":
            history.clear()
            print("History cleared\n")
            continue

        history.append({"role": "user", "content": user_input})
        history = history[-MAX_HISTORY:]

        reply = backend.generate_reply(user_text=user_input, messages=history, memory_k=MAX_HISTORY)
        final = reply.answer

        print("\n--- ANSWER ---")
        print(final)

        print("\n--- SOURCES ---")
        print(reply.sources_text)

        history.append({"role": "assistant", "content": final})


if __name__ == "__main__":
    main()
