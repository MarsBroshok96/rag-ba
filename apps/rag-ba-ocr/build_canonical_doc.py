from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def sort_regions_reading_order(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Простая эвристика reading order:
    1. Сортируем по top (y0)
    2. Если близко по y — сортируем по x0
    """
    def key(r: dict[str, Any]):
        x0, y0, x1, y1 = r["bbox_px"]
        return (y0, x0)

    return sorted(regions, key=key)


def merge_lines_to_text(lines: list[dict[str, Any]]) -> str:
    """
    Склеиваем строки региона в абзац.
    """
    if not lines:
        return ""

    # сортируем строки внутри региона по y
    lines = sorted(
        lines,
        key=lambda x: (
            (x.get("bbox_px") or [0, 0, 0, 0])[1],  # y0
            (x.get("bbox_px") or [0, 0, 0, 0])[0],  # x0
        ),
        )

    texts = []
    for l in lines:
        txt = l.get("text", "").strip()
        if txt:
            texts.append(txt)

    # простая склейка с пробелом
    return " ".join(texts)


def build_page_document(page_json_path: Path) -> dict[str, Any]:
    data = json.loads(page_json_path.read_text(encoding="utf-8"))
    regions = data.get("regions", [])

    # фильтруем только текстовые регионы
    text_regions = [
        r for r in regions
        if r.get("type") in ("text", "title", "list", "table")
    ]

    text_regions = sort_regions_reading_order(text_regions)

    blocks = []
    for r in text_regions:
        paragraph = merge_lines_to_text(r.get("ocr_lines", []))

        blocks.append(
            {
                "region_id": r.get("id"),
                "type": r.get("type"),
                "bbox_px": r.get("bbox_px"),
                "text": paragraph,
                # полезные поля для трассировки
                "crop_path": r.get("crop_path"),
                "page_image_path": r.get("page_image_path"),
            }
        )

    return {
        "doc_id": data.get("doc_id"),
        "source_pdf": data.get("source_pdf"),
        "page": data.get("page"),
        "image": data.get("image"),     # путь к page png (если есть)
        "blocks": blocks,
    }


def main() -> None:
    input_root = Path("layout_ocr")
    output_root = Path("canon")
    output_root.mkdir(exist_ok=True)

    assert input_root.exists(), f"Missing input dir: {input_root}"

    # ожидаем layout_ocr/<doc_id>/pageXXX_layout_ocr.json
    doc_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not doc_dirs:
        raise RuntimeError(f"No doc folders found in {input_root} (expected layout_ocr/<doc_id>/...)")

    for doc_dir in doc_dirs:
        doc_id = doc_dir.name
        out_dir = output_root / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        page_jsons = sorted(doc_dir.glob("page*_layout_ocr.json"))
        if not page_jsons:
            print(f"Skip doc {doc_id}: no page*_layout_ocr.json found")
            continue

        print(f"\nDOC: {doc_id} pages={len(page_jsons)}")

        for page_json_path in page_jsons:
            doc = build_page_document(page_json_path)

            # номер страницы — из json (если есть), иначе пытаемся вытащить из имени файла
            page = doc.get("page")
            if not isinstance(page, int):
                # page001_layout_ocr.json
                stem = page_json_path.stem
                try:
                    page = int(stem.split("_")[0].replace("page", ""))
                except Exception:
                    page = 0

            out_path = out_dir / f"page{int(page):03d}_canonical.json"
            out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"  Built canonical page {int(page):03d} -> {out_path.relative_to(output_root)}")

    print("\nDone.")
    

if __name__ == "__main__":
    main()
