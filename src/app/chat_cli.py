from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.index.test_query_with_citations import load_manifest, format_sources


MAX_HISTORY = 6


def build_index(project_root: Path):
    persist_dir = project_root / "data" / "vectorstore" / "chroma_rag"
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name="rag")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    project_root = Path(__file__).resolve().parents[2]

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base"
    )
    Settings.llm = Ollama(model="qwen2.5:14b-instruct", request_timeout=180.0)

    manifest = load_manifest(project_root)
    index = build_index(project_root)

    query_engine = index.as_query_engine(similarity_top_k=8)

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

        # Retrieve
        resp = query_engine.query(user_input)

        sources_block_llm = format_sources(
            resp,
            project_root=project_root,
            manifest=manifest,
            max_sources=6,
            snippet_chars=700,
            include_paths=False,
        )

        conversation_context = "\n".join(
            f"{m['role']}: {m['content']}" for m in history
        )

        prompt = f"""
You are answering using RAG.

Conversation history:
{conversation_context}

Rules:
- Use ONLY the information explicitly present in the SNIPPET texts below.
- Every bullet/claim must end with citations like [1] or [2][3].
- Do NOT introduce new concepts not present in snippets (e.g. "regulatory compliance") unless those words/ideas appear in snippets.
- If the snippets are insufficient for some part of the question, write: "NOT FOUND IN DOCUMENTS" and stop. No hypothesis unless explicitly requested..

Sources:
{sources_block_llm}

Answer:
"""

        final = Settings.llm.complete(prompt).text

        print("\n--- ANSWER ---")
        print(final)

        print("\n--- SOURCES ---")
        print(
            format_sources(
                resp,
                project_root=project_root,
                manifest=manifest,
                max_sources=6,
                snippet_chars=200,
                include_paths=True,
            )
        )

        history.append({"role": "assistant", "content": final})


if __name__ == "__main__":
    main()