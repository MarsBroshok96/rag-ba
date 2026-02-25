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
    assert "score=0.9123" in out
    assert "SNIPPET: line one line two" in out
