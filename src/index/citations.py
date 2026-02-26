from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(project_root: Path) -> dict[str, Any]:
    path = project_root / "data" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
    _ = project_root  # kept for stable call signature / compatibility
    doc_id = md.get("doc_id")
    page = md.get("page")
    if not doc_id:
        return None

    doc = (manifest.get("docs") or {}).get(doc_id) or {}
    source_path = doc.get("source_path")

    crop_file = md.get("crop_file")
    if not crop_file:
        cp = md.get("crop_path")
        if isinstance(cp, str) and cp.strip():
            crop_file = Path(cp).name

    if crop_file and page:
        crop_dir_tpl = doc.get("crop_dir_tpl")
        if isinstance(crop_dir_tpl, str) and crop_dir_tpl:
            crop_dir = Path(crop_dir_tpl.format(page=int(page)))
            p = (crop_dir / str(crop_file)).resolve()
            if p.exists():
                return p

    if page:
        page_tpl = doc.get("page_image_tpl")
        if isinstance(page_tpl, str) and page_tpl:
            p = Path(page_tpl.format(page=int(page))).resolve()
            if p.exists():
                return p

    if isinstance(source_path, str) and source_path:
        p = Path(source_path).resolve()
        if p.exists():
            return p

    return None


def resolve_source_document_path(
    *,
    manifest: dict[str, Any],
    md: dict[str, Any],
) -> Path | None:
    doc_id = md.get("doc_id")
    if not doc_id:
        return None
    doc = (manifest.get("docs") or {}).get(doc_id) or {}
    source_path = doc.get("source_path")
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
    only_source_ids: set[int] | None = None,
) -> str:
    def _snip(s: str, n: int) -> str:
        s = (s or "").replace("\n", " ").strip()
        return (s[:n] + "…") if len(s) > n else s

    lines: list[str] = []
    for i, sn in enumerate(getattr(resp, "source_nodes", [])[:max_sources], start=1):
        if only_source_ids is not None and i not in only_source_ids:
            continue
        md = sn.node.metadata or {}

        rid = md.get("region_id") or md.get("id") or md.get("chunk_id")
        src = md.get("source_file")
        page = md.get("page")
        typ = md.get("type")

        score = getattr(sn, "score", None)
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        txt = _snip(sn.node.get_text() or "", snippet_chars)

        best_path = resolve_best_path(manifest=manifest, project_root=project_root, md=md)
        best_uri = None
        if best_path:
            try:
                best_uri = best_path.as_uri()
            except Exception:
                best_uri = None

        doc_path = resolve_source_document_path(manifest=manifest, md=md)
        doc_uri = None
        if doc_path:
            try:
                doc_uri = doc_path.as_uri()
            except Exception:
                doc_uri = None

        head = f"[{i}] score={score_s} file={src} page={page} type={typ} region={rid}"

        path_line = ""
        if include_paths and best_path:
            path_line = f"    PATH: {best_path}"
            if best_uri:
                path_line += f"\n    URI:  {best_uri}"
        if include_paths and doc_path:
            doc_lines = f"    DOC_PATH: {doc_path}"
            if doc_uri:
                doc_lines += f"\n    DOC_URI:  {doc_uri}"
            path_line = (path_line + "\n" if path_line else "") + doc_lines

        lines.append(head + ("\n" + path_line if path_line else "") + f"\n    SNIPPET: {txt}")

    return "\n\n".join(lines) if lines else "(no sources)"
