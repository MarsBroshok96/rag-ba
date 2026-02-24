from __future__ import annotations

import os
from pathlib import Path

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    project_root = Path(__file__).resolve().parents[2]
    persist_dir = project_root / "data" / "vectorstore" / "chroma_rag"

    assert persist_dir.exists(), f"Vector DB not found: {persist_dir}"

    # Embeddings — должны совпадать с теми, что использовались при индексации
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base"
    )

    # Локальная LLM через Ollama
    Settings.llm = Ollama(
        model="qwen2.5:14b-instruct",
        request_timeout=180.0,
    )

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name="rag")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    query_engine = index.as_query_engine(similarity_top_k=5)

    q = "What are the main challenges in Chemical Product Engineering (CPE) discussed in the document?"
    response = query_engine.query(q)

    print("\n=== QUESTION ===")
    print(q)

    print("\n=== ANSWER ===")
    print(response)

    print("\n=== SOURCES ===")
    for i, sn in enumerate(response.source_nodes[:5], start=1):
        md = sn.node.metadata or {}
        text_snippet = sn.node.get_text().replace("\n", " ")[:240]
        print(
            f"{i}. score={sn.score:.4f} "
            f"file={md.get('source_file')} "
            f"page={md.get('page')} "
            f"type={md.get('type')}"
        )
        print(f"   {text_snippet}...")
    

if __name__ == "__main__":
    main()
