from __future__ import annotations

import os
import shutil
from pathlib import Path

import chromadb
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


def main() -> None:
    """
    Health check:
    1) Create local Chroma persistent store
    2) Build index
    3) Query via local Ollama LLM (no OpenAI)
    """

    project_root = Path(__file__).resolve().parents[2]
    persist_dir = project_root / "data" / "vectorstore" / "chroma_demo"

    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Local embeddings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base"
    )

    # LOCAL LLM via Ollama (critical line)
    Settings.llm = Ollama(
        model="qwen2.5:14b-instruct",
        request_timeout=120.0,
    )

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name="demo")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = [
        Document(
            text=(
                "RAG system requirement: answers must be grounded in documents with citations. "
                "If information is missing, label it explicitly as a HYPOTHESIS."
            ),
            metadata={"source_file": "demo_req.txt", "page": 1, "paragraph": 1},
        ),
        Document(
            text=(
                "Project constraints: avoid data leaks to the internet by default. "
                "Use local LLM whenever possible; cloud is optional and must be explicit."
            ),
            metadata={"source_file": "demo_req.txt", "page": 1, "paragraph": 2},
        ),
    ]

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True,
    )

    query_engine = index.as_query_engine(similarity_top_k=3)

    q = "Какие требования к ответам и как обрабатывать отсутствие информации?"
    resp = query_engine.query(q)

    print("\n=== QUESTION ===")
    print(q)

    print("\n=== ANSWER ===")
    print(resp)

    print("\n=== SOURCES ===")
    for i, sn in enumerate(getattr(resp, "source_nodes", [])[:5], start=1):
        text_snippet = (sn.node.get_text() or "").replace("\n", " ").strip()
        text_snippet = text_snippet[:240] + ("…" if len(text_snippet) > 240 else "")
        md = sn.node.metadata or {}
        print(
            f"{i}. score={sn.score:.4f} "
            f"source={md.get('source_file')} "
            f"page={md.get('page')} "
            f"paragraph={md.get('paragraph')}"
        )
        print(f"   {text_snippet}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
