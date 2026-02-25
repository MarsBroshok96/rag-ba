from __future__ import annotations

from src.common import project_paths as p


def test_project_paths_are_repo_local() -> None:
    assert p.PROJECT_ROOT.name == "rag-ba-agents"
    assert p.DATA_DIR == p.PROJECT_ROOT / "data"
    assert p.RAG_BA_OCR_DIR == p.PROJECT_ROOT / "apps" / "rag-ba-ocr"
    assert p.OCR_EXPORT_ROOT == p.RAG_BA_OCR_DIR / "export"


def test_vectorstore_paths_share_same_parent() -> None:
    assert p.CHROMA_RAG_DIR.parent == p.VECTORSTORE_ROOT
    assert p.CHROMA_DEMO_DIR.parent == p.VECTORSTORE_ROOT
