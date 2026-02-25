from __future__ import annotations

import os


import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.common.project_paths import CHROMA_RAG_DIR, PROJECT_ROOT
from src.index.citations import format_sources, load_manifest

#QUESTION = "What are the main challenges in Chemical Product Engineering (CPE) discussed in the document?"
#QUESTION = "Что такое системы RTO и как она взаимодействует с СУУТП?"
QUESTION = "What is ML?"


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    project_root = PROJECT_ROOT
    manifest = load_manifest(project_root)
    persist_dir = CHROMA_RAG_DIR
    assert persist_dir.exists(), f"Vector DB not found: {persist_dir}"

    # must match indexing
    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
    Settings.llm = Ollama(model="qwen2.5:14b-instruct", request_timeout=180.0)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name="rag")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    query_engine = index.as_query_engine(similarity_top_k=6)

    resp = query_engine.query(QUESTION)

    sources_block_llm = format_sources(
        resp,
        project_root=project_root,
        manifest=manifest,
        max_sources=8,
        snippet_chars=700,
        include_paths=False,  # LLM-у пути не нужны (и могут отвлекать)
    )

    sources_block_print = format_sources(
        resp,
        project_root=project_root,
        manifest=manifest,
        max_sources=8,
        snippet_chars=200,
        include_paths=True,   # для консоли хотим кликабельность
    )

    prompt = f"""You are answering using RAG.
Rules:
- Use ONLY the information explicitly present in the SNIPPET texts below.
- Every bullet/claim must end with citations like [1] or [2][3].
- Do NOT introduce new concepts not present in snippets (e.g. "regulatory compliance") unless those words/ideas appear in snippets.
- If the snippets are insufficient for some part of the question, write: "NOT FOUND IN DOCUMENTS" and stop. No hypothesis unless explicitly requested.


Question:
{QUESTION}

Sources list (for citations):
{sources_block_llm}

Now write the answer with citations:
"""

    final = Settings.llm.complete(prompt).text

    print("\n=== QUESTION ===")
    print(QUESTION)

    print("\n=== ANSWER (with citations) ===")
    print(final)

    print("\n=== SOURCES (numbered) ===")
    print(sources_block_print)



if __name__ == "__main__":
    main()
