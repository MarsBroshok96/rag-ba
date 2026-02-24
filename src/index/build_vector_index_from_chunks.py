from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable
import hashlib

import chromadb
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def _iter_chunks(obj: Any) -> Iterable[dict[str, Any]]:
    """
    Поддерживаем разные форматы:
    - list[chunk]
    - {"chunks": [...]}
    - {"items": [...]}
    """
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
        return
    if isinstance(obj, dict):
        for key in ("chunks", "items", "data"):
            v = obj.get(key)
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        yield x
                return
    raise ValueError("Unknown chunks.json format: expected list or dict with chunks/items/data")


def _chunk_text(ch: dict[str, Any]) -> str:
    # типичные варианты ключей
    for k in ("text", "chunk", "content"):
        v = ch.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _to_rel(project_root: Path, p: str | None) -> str | None:
    if not p or not isinstance(p, str):
        return None
    try:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp.relative_to(project_root))
        return p
    except Exception:
        # если relative_to не получилось — вернем как есть (но лучше не абсолютный)
        return p


def _slim_metadata(md: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """
    Делаем metadata гарантированно короткой (<~300-500 символов).
    В индексе храним только то, что нужно для навигации/цитат.
    """
    out: dict[str, Any] = {}

    # MUST-HAVE для цитирования
    for k in ("chunk_id", "doc_id", "source_file", "page", "type"):
        v = md.get(k)
        if v is not None:
            out[k] = v

    # region_ids: оставляем максимум 1 (или вообще убери)
    # Это ключевой источник раздувания metadata.
    rids = md.get("region_ids")
    if isinstance(rids, list) and rids:
        out["region_id"] = str(rids[0])  # только первый
    elif isinstance(rids, str) and rids:
        out["region_id"] = rids.split(",")[0].strip()

    # crop_path: только basename (путь длинный и не нужен в индексе)
    cp = None
    cps = md.get("crop_paths")
    if isinstance(cps, list) and cps:
        cp = str(cps[0])
    elif isinstance(md.get("crop_path"), str):
        cp = md.get("crop_path")

    if isinstance(cp, str) and cp:
        out["crop_file"] = Path(cp).name

    # pdf/page_image: тоже только basename
    pimg = md.get("page_image_path")
    if isinstance(pimg, str) and pimg:
        out["page_image_file"] = Path(pimg).name

    pdf = md.get("source_path")
    if isinstance(pdf, str) and pdf:
        out["pdf_file"] = Path(pdf).name

    # chunks_json_path выкидываем полностью
    return out


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _rel_or_abs(project_root: Path, p: str | None) -> str | None:
    if not p or not isinstance(p, str):
        return None
    pp = Path(p)
    try:
        if pp.is_absolute():
            # храним относительный (если можем), иначе абсолютный
            try:
                return str(pp.relative_to(project_root))
            except Exception:
                return str(pp)
        return p
    except Exception:
        return p


def _build_manifest_for_doc(
    *,
    project_root: Path,
    rag_ba_ocr_root: Path,
    doc_id: str,
    source_path: str | None,
) -> dict[str, Any]:
    """
    Manifest хранит “истину” о путях, чтобы не засорять metadata в индексе.
    """
    layout_dir = (project_root / "layout" / doc_id).resolve()
    ocr_dir = (rag_ba_ocr_root / "layout_ocr" / doc_id).resolve()

    entry: dict[str, Any] = {
        "doc_id": doc_id,
        "layout_dir": str(layout_dir),
        "ocr_dir": str(ocr_dir),
        "page_image_tpl": str(layout_dir / "page{page:03d}.png"),
        "crop_dir_tpl": str(ocr_dir / "page{page:03d}_crops"),
    }

    if isinstance(source_path, str) and source_path.strip():
        entry["source_path"] = str(Path(source_path).resolve())
        entry["source_file"] = Path(source_path).name
        entry["doc_fingerprint"] = _sha1(entry["source_path"])
    else:
        entry["source_path"] = None
        entry["source_file"] = None
        entry["doc_fingerprint"] = None

    return entry

def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    project_root = Path(__file__).resolve().parents[2]

    # Берем chunks.json из соседнего репо (теперь: export/<doc_id>/chunks.json)
    export_root = (project_root.parent / "rag-ba" / "apps" / "rag-ba-ocr" / "export").resolve()
    
    rag_ba_ocr_root = (project_root.parent / "rag-ba" / "apps" / "rag-ba-ocr").resolve()
    
    assert export_root.exists(), f"Missing export dir: {export_root}"

    chunks_paths = sorted(export_root.glob("*/chunks.json"))
    assert chunks_paths, f"No chunks.json found under: {export_root} (expected export/<doc_id>/chunks.json)"

    # Папка хранилища Chroma ВНУТРИ rag-ba (чтобы не путаться)
    persist_dir = project_root / "data" / "vectorstore" / "chroma_rag"
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Embeddings (как в healthcheck)
    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

    docs: list[Document] = []
    loaded_files = 0
    total_chunks_seen = 0
    
    manifest: dict[str, Any] = {
        "version": 1,
        "generated_by": "build_vector_index_from_chunks.py",
        "docs": {},  # doc_id -> entry
    }

    for chunks_path in chunks_paths:
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        
         # --- manifest per doc ---
        doc_id = chunks_path.parent.name

        # попробуем вытащить source_path
        source_path: str | None = None
        if isinstance(data, list) and data:
            maybe = data[0].get("pdf_path") or data[0].get("source_path")
            if isinstance(maybe, str) and maybe.strip():
                source_path = maybe

        if doc_id not in manifest["docs"]:
            manifest["docs"][doc_id] = _build_manifest_for_doc(
                project_root=project_root,
                rag_ba_ocr_root=rag_ba_ocr_root,
                doc_id=doc_id,
                source_path=source_path,
            )
            
        loaded_files += 1

        for ch in _iter_chunks(data):
            total_chunks_seen += 1
            text = _chunk_text(ch)
            if not text:
                continue

            md = dict(ch.get("metadata") or {})
            md.setdefault("chunks_json_path", str(chunks_path.resolve()))

            # Метаданные: берём всё полезное, что есть в нашем chunks.json
            for k in (
                "source_file",
                "doc_id",
                "page",
                "type",
                "source_path",
                "page_image_path",
                "chunk_id",
                "region_ids",
                "crop_paths",
            ):
                if k in ch and k not in md:
                    md[k] = ch[k]

            # Совместимость: иногда upstream мог назвать по-другому
            if "region_id" in ch and "region_id" not in md:
                md["region_id"] = ch["region_id"]
            if "crop_path" in ch and "crop_path" not in md:
                md["crop_path"] = ch["crop_path"]

            # chunk_id: у нас он уже есть (doc__cN). Если нет — сделаем стабильный от файла+счётчика
            md.setdefault("chunk_id", ch.get("chunk_id") or f"{chunks_path.parent.name}__chunk_{total_chunks_seen}")
            md.setdefault("source_file", md.get("source_file") or md.get("doc_id") or "unknown")
            
            md = _slim_metadata(md, project_root)
            
            docs.append(Document(text=text, metadata=md))

    print(f"Loaded docs: {len(docs)} from {loaded_files} chunks.json files under {export_root}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name="rag")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    _ = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"Indexed {len(docs)} chunks into Chroma")
    print(f"Vector DB stored in {persist_dir}")
    
    manifest_path = project_root / "data" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest stored in {manifest_path}")


if __name__ == "__main__":
    main()
