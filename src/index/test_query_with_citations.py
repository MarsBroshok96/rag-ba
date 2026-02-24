from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import json


import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

#QUESTION = "What are the main challenges in Chemical Product Engineering (CPE) discussed in the document?"
#QUESTION = "Что такое системы RTO и как она взаимодействует с СУУТП?"
QUESTION = "What is ML?"


def load_manifest(project_root: Path) -> dict[str, Any]:
    p = project_root / "data" / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_best_path(
    *,
    manifest: dict[str, Any],
    project_root: Path,
    md: dict[str, Any],
) -> Path | None:
    """
    Выбираем лучший путь для клика: crop -> page image -> pdf.
    Пути берём из manifest, а из metadata — только doc_id/page + basename’ы.
    """
    doc_id = md.get("doc_id")
    page = md.get("page")
    if not doc_id:
        return None

    doc = (manifest.get("docs") or {}).get(doc_id) or {}
    source_path = doc.get("source_path")

    # crop
    crop_file = md.get("crop_file")
    if not crop_file:
        cp = md.get("crop_path")
        if isinstance(cp, str) and cp.strip():
            crop_file = Path(cp).name  # берём только имя файла
    
    crop_file = md.get("crop_file")
    if crop_file and page:
        crop_dir_tpl = doc.get("crop_dir_tpl")
        if isinstance(crop_dir_tpl, str) and crop_dir_tpl:
            crop_dir = Path(crop_dir_tpl.format(page=int(page)))
            p = (crop_dir / str(crop_file)).resolve()
            if p.exists():
                return p

    # page image
    if page:
        page_tpl = doc.get("page_image_tpl")
        if isinstance(page_tpl, str) and page_tpl:
            p = Path(page_tpl.format(page=int(page))).resolve()
            if p.exists():
                return p

    # pdf
    if isinstance(source_path, str) and source_path:
        p = Path(source_path).resolve()
        if p.exists():
            return p

    return None


def format_sources(
    resp: Any,
    project_root: Path,
    manifest: dict[str, Any],
    max_sources: int = 8,
    snippet_chars: int = 700,
    include_paths: bool = True,
) -> str:
    def _snip(s: str, n: int) -> str:
        s = (s or "").replace("\n", " ").strip()
        return (s[:n] + "…") if len(s) > n else s

    lines: list[str] = []
    for i, sn in enumerate(getattr(resp, "source_nodes", [])[:max_sources], start=1):
        md = sn.node.metadata or {}

        rid = md.get("region_id") or md.get("id") or md.get("chunk_id")
        src = md.get("source_file")
        page = md.get("page")
        typ = md.get("type")

        score = getattr(sn, "score", None)
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"

        # текст — только для консоли (не влияет на retrieval)
        txt = _snip(sn.node.get_text() or "", snippet_chars)

        best_path = resolve_best_path(manifest=manifest, project_root=project_root, md=md)
        uri = None
        if best_path:
            try:
                uri = best_path.as_uri()
            except Exception:
                uri = None

        head = f"[{i}] score={score_s} file={src} page={page} type={typ} region={rid}"

        path_line = ""
        if include_paths and best_path:
            path_line = f"    PATH: {best_path}"
            if uri:
                path_line += f"\n    URI:  {uri}"

        lines.append(
            head
            + ("\n" + path_line if path_line else "")
            + f"\n    SNIPPET: {txt}"
        )

    return "\n\n".join(lines) if lines else "(no sources)"


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    project_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(project_root)
    persist_dir = project_root / "data" / "vectorstore" / "chroma_rag"
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
