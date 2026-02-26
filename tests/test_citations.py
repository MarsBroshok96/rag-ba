from __future__ import annotations

from pathlib import Path

from src.index.citations import format_sources, resolve_best_path


class _FakeNode:
    def __init__(self, text: str, metadata: dict):
        self._text = text
        self.metadata = metadata

    def get_text(self) -> str:
        return self._text


class _FakeSourceNode:
    def __init__(self, text: str, metadata: dict, score: float = 0.5):
        self.node = _FakeNode(text, metadata)
        self.score = score


class _FakeResp:
    def __init__(self, nodes: list[_FakeSourceNode]):
        self.source_nodes = nodes


def test_resolve_best_path_prefers_crop_and_fallback_crop_path_basename(tmp_path: Path) -> None:
    doc_dir = tmp_path / "doc1"
    crop_dir = doc_dir / "page001_crops"
    crop_dir.mkdir(parents=True)
    crop_file = crop_dir / "r1_text.png"
    crop_file.write_bytes(b"x")

    page_image = doc_dir / "page001.png"
    page_image.write_bytes(b"x")
    pdf_file = tmp_path / "source.pdf"
    pdf_file.write_bytes(b"x")

    manifest = {
        "docs": {
            "doc1": {
                "crop_dir_tpl": str(doc_dir / "page{page:03d}_crops"),
                "page_image_tpl": str(doc_dir / "page{page:03d}.png"),
                "source_path": str(pdf_file),
            }
        }
    }

    md = {
        "doc_id": "doc1",
        "page": 1,
        "crop_path": "/some/other/place/r1_text.png",
    }
    got = resolve_best_path(manifest=manifest, project_root=tmp_path, md=md)
    assert got == crop_file.resolve()


def test_format_sources_includes_clickable_path_when_present(tmp_path: Path) -> None:
    page_image = tmp_path / "page001.png"
    page_image.write_bytes(b"x")

    manifest = {
        "docs": {
            "docA": {
                "page_image_tpl": str(tmp_path / "page{page:03d}.png"),
                "crop_dir_tpl": str(tmp_path / "page{page:03d}_crops"),
                "source_path": str(tmp_path / "docA.pdf"),
            }
        }
    }
    (tmp_path / "docA.pdf").write_bytes(b"x")

    resp = _FakeResp(
        [
            _FakeSourceNode(
                "line one\nline two",
                {
                    "doc_id": "docA",
                    "page": 1,
                    "source_file": "docA.pdf",
                    "type": "text",
                    "chunk_id": "docA__c1",
                },
                score=0.9123,
            )
        ]
    )

    out = format_sources(resp, project_root=tmp_path, manifest=manifest, include_paths=True, snippet_chars=20)
    assert "PATH:" in out
    assert "URI:" in out
    assert "DOC_PATH:" in out
    assert "DOC_URI:" in out
    assert "score=0.9123" in out
    assert "SNIPPET: line one line two" in out


def test_format_sources_can_filter_by_citation_ids_preserving_labels(tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").write_bytes(b"x")
    (tmp_path / "page001.png").write_bytes(b"x")
    (tmp_path / "page002.png").write_bytes(b"x")
    (tmp_path / "page003.png").write_bytes(b"x")

    manifest = {
        "docs": {
            "docA": {
                "page_image_tpl": str(tmp_path / "page{page:03d}.png"),
                "crop_dir_tpl": str(tmp_path / "page{page:03d}_crops"),
                "source_path": str(tmp_path / "doc.pdf"),
            }
        }
    }

    resp = _FakeResp(
        [
            _FakeSourceNode("text 1", {"doc_id": "docA", "page": 1, "chunk_id": "c1"}),
            _FakeSourceNode("text 2", {"doc_id": "docA", "page": 2, "chunk_id": "c2"}),
            _FakeSourceNode("text 3", {"doc_id": "docA", "page": 3, "chunk_id": "c3"}),
        ]
    )

    out = format_sources(
        resp,
        project_root=tmp_path,
        manifest=manifest,
        include_paths=False,
        only_source_ids={2, 3},
    )

    assert "[1]" not in out
    assert "[2]" in out
    assert "[3]" in out
    assert "text 2" in out
    assert "text 3" in out
