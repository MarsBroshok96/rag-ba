from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MAX_CHARS = 1200
MIN_CHARS = 400


def _page_image_path(doc_id: str, page_num: int, block: dict[str, Any]) -> str:
    # prefer exact path from upstream (canonical/full_document)
    p = block.get("page_image_path")
    if isinstance(p, str) and p.strip():
        return p

    # fallback: rag-ba layout convention
    return str(Path("layout") / doc_id / f"page{int(page_num):03d}.png")


def flush_chunk(chunks: list[dict], current: dict | None) -> None:
    if not current:
        return
    if current["text"].strip():
        current["char_len"] = len(current["text"])
        chunks.append(current)


def main() -> None:
    export_root = Path("export")
    assert export_root.exists(), f"Missing export dir: {export_root}"

    doc_dirs = sorted([p for p in export_root.iterdir() if p.is_dir()])
    if not doc_dirs:
        raise RuntimeError(f"No doc folders found in {export_root} (expected export/<doc_id>/...)")

    for doc_dir in doc_dirs:
        in_path = doc_dir / "full_document.json"
        if not in_path.exists():
            print(f"Skip {doc_dir.name}: missing {in_path.name}")
            continue

        out_path = doc_dir / "chunks.json"

        data = json.loads(in_path.read_text(encoding="utf-8"))
#        pages = data.get("pages", [])
        # --- Universal handling: PDF (pages[]) OR DOCX (single page blocks) ---
        if "pages" in data and isinstance(data["pages"], list):
            pages = data["pages"]
        elif "blocks" in data and isinstance(data["blocks"], list):
            # single-page doc (e.g. docx)
            pages = [
                {
                    "page": data.get("page", 1),
                    "blocks": data["blocks"],
                }
            ]
        else:
            pages = []

        doc_id = data.get("doc_id", doc_dir.name)
        source_path = data.get("source_pdf") or data.get("source_path")
        source_file = data.get("source_file") or (
            Path(source_path).name if isinstance(source_path, str) else "unknown"
        )


        chunks: list[dict[str, Any]] = []
        chunk_id = 1
        current: dict[str, Any] | None = None

        for page in pages:
            page_num = page.get("page")
            blocks = page.get("blocks", [])

            for b in blocks:
                typ = b.get("type")
                text = (b.get("text") or "").strip()

                if not text:
                    continue

                if typ == "figure":
                    continue

                if typ == "title":
                    flush_chunk(chunks, current)
                    current = {
                        "chunk_id": f"{doc_id}__c{chunk_id}",
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "source_path": source_path,
                        "page": int(page_num) if page_num is not None else None,
                        "page_image_path": _page_image_path(doc_id, int(page_num), b),
                        "type": "title",
                        "region_ids": [b.get("region_id")],
                        "crop_paths": ([b["crop_path"]] if isinstance(b.get("crop_path"), str) else []),
                        "text": text + "\n",
                    }
                    chunk_id += 1
                    continue

                if current is None:
                    current = {
                        "chunk_id": f"{doc_id}__c{chunk_id}",
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "source_path": source_path,
                        "page": int(page_num) if page_num is not None else None,
                        "page_image_path": _page_image_path(doc_id, int(page_num), b),
                        "type": typ,
                        "region_ids": [],
                        "crop_paths": [],
                        "text": "",
                    }
                    chunk_id += 1

                if len(current["text"]) + len(text) > MAX_CHARS:
                    flush_chunk(chunks, current)
                    current = {
                        "chunk_id": f"{doc_id}__c{chunk_id}",
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "source_path": source_path,
                        "page": int(page_num) if page_num is not None else None,
                        "page_image_path": _page_image_path(doc_id, int(page_num), b),
                        "type": typ,
                        "region_ids": [],
                        "crop_paths": [],
                        "text": "",
                    }
                    chunk_id += 1

                current["text"] += text + "\n"

                rid = b.get("region_id")
                if rid:
                    current["region_ids"].append(rid)

                cp = b.get("crop_path")
                if isinstance(cp, str) and cp.strip():
                    current.setdefault("crop_paths", []).append(cp)

        flush_chunk(chunks, current)

        out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DOC {doc_id}: saved {len(chunks)} chunks → {out_path}")


if __name__ == "__main__":
    main()
