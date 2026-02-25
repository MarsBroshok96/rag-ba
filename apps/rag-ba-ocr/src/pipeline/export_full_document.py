from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lxml import etree

from src.common.project_paths import OCR_CANON_ROOT, OCR_EXPORT_ROOT


def region_sort_key(b: dict[str, Any]) -> tuple[int, int]:
    # Сортировка блоков внутри страницы: сверху вниз, затем слева направо
    x0, y0, x1, y1 = b.get("bbox_px", [0, 0, 0, 0])
    return (int(y0), int(x0))


def build_full_doc(canon_doc_dir: Path, pages: list[int] | None = None) -> dict[str, Any]:
    files = sorted(canon_doc_dir.glob("page*_canonical.json"))

    if pages is not None:
        wanted = {int(p) for p in pages}
        files = [
            f for f in files
            if int(f.stem.split("_")[0].replace("page", "")) in wanted
        ]

    doc_pages: list[dict[str, Any]] = []
    doc_id: str | None = None
    source_pdf: str | None = None

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))

        # один раз снимаем метаданные документа (они одинаковые на всех страницах)
        if doc_id is None:
            doc_id = data.get("doc_id")
        if source_pdf is None:
            source_pdf = data.get("source_pdf")

        page_num = int(data.get("page", 0))
        blocks = data.get("blocks", [])
        blocks = sorted(blocks, key=region_sort_key)

        doc_pages.append({"page": page_num, "blocks": blocks})

    return {
        "doc_id": doc_id,
        "source_pdf": source_pdf,
        "page_count": len(doc_pages),
        "pages": doc_pages,
    }


def export_json(full_doc: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(full_doc, ensure_ascii=False, indent=2), encoding="utf-8")


def export_xml(full_doc: dict[str, Any], out_path: Path) -> None:
    root = etree.Element("document")
    etree.SubElement(root, "doc_id").text = str(full_doc.get("doc_id", "") or "")
    etree.SubElement(root, "source_pdf").text = str(full_doc.get("source_pdf", "") or "")

    pages_el = etree.SubElement(root, "pages")
    for p in full_doc.get("pages", []):
        page_el = etree.SubElement(pages_el, "page", number=str(p.get("page", "")))
        for b in p.get("blocks", []):
            attrs = {
                "region_id": str(b.get("region_id", "")),
                "type": str(b.get("type", "")),
            }
            if b.get("crop_path"):
                attrs["crop_path"] = str(b.get("crop_path"))
            if b.get("page_image_path"):
                attrs["page_image_path"] = str(b.get("page_image_path"))
                
            bbox = b.get("bbox_px")
            if bbox and isinstance(bbox, list) and len(bbox) == 4:
                attrs["x0"] = str(bbox[0])
                attrs["y0"] = str(bbox[1])
                attrs["x1"] = str(bbox[2])
                attrs["y1"] = str(bbox[3])

            block_el = etree.SubElement(page_el, "block", **attrs)
            text = b.get("text", "")
            block_el.text = text

    tree = etree.ElementTree(root)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True, pretty_print=True)


def export_flat_text(full_doc: dict[str, Any], out_path: Path) -> None:
    # Быстрый sanity-check: текст по страницам
    lines: list[str] = []
    for p in full_doc.get("pages", []):
        lines.append(f"\n=== PAGE {p.get('page')} ===\n")
        for b in p.get("blocks", []):
            typ = b.get("type", "unknown")
            rid = b.get("region_id", "")
            txt = (b.get("text", "") or "").strip()
            if not txt:
                continue
            lines.append(f"[{typ}] ({rid}) {txt}\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    in_root = OCR_CANON_ROOT
    out_root = OCR_EXPORT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    assert in_root.exists(), f"Missing canon dir: {in_root}"

    doc_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])
    if not doc_dirs:
        raise RuntimeError(f"No doc folders found in {in_root} (expected canon/<doc_id>/...)")

    for doc_dir in doc_dirs:
        doc_id = doc_dir.name
        out_dir = out_root / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        full_doc = build_full_doc(doc_dir, pages=None)

        export_json(full_doc, out_dir / "full_document.json")
        export_xml(full_doc, out_dir / "full_document.xml")
        export_flat_text(full_doc, out_dir / "full_document.txt")

        print(f"\nDOC: {doc_id}")
        print(f"  Saved: {out_dir/'full_document.json'}")
        print(f"  Saved: {out_dir/'full_document.xml'}")
        print(f"  Saved: {out_dir/'full_document.txt'}")
        print(f"  Pages exported: {full_doc.get('page_count')}")


if __name__ == "__main__":
    main()
