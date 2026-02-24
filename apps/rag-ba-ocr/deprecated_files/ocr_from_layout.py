from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image


@dataclass(frozen=True)
class Region:
    id: str
    type: str  # text/table/figure/sidebar/other
    page: int
    bbox_norm: tuple[float, float, float, float]  # x0,y0,x1,y1 in [0,1]


def to_jsonable(obj: Any) -> Any:
    import numpy as _np

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return to_jsonable(vars(obj))
    return str(obj)


def deep_get_text_items(root: Any) -> list[dict]:
    items: list[dict] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            texts = x.get("rec_texts") or x.get("texts")
            scores = x.get("rec_scores") or x.get("scores")
            boxes = x.get("dt_polys") or x.get("boxes") or x.get("polys") or x.get("det_polys")

            if isinstance(texts, list) and isinstance(scores, list):
                n = min(len(texts), len(scores), len(boxes) if isinstance(boxes, list) else len(texts))
                for i in range(n):
                    items.append(
                        {
                            "text": str(texts[i]),
                            "score": float(scores[i]),
                            "box": to_jsonable(boxes[i]) if isinstance(boxes, list) else None,
                        }
                    )

            for v in x.values():
                walk(v)
            return

        if isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
            return

        if hasattr(x, "__dict__"):
            walk(vars(x))
            return

    walk(root)

    # de-dupe
    seen = set()
    uniq: list[dict] = []
    for it in items:
        key = (it.get("text"), round(float(it.get("score", 0.0)), 6), json.dumps(it.get("box"), ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    return uniq


def load_regions(layout_path: Path) -> list[Region]:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    regions: list[Region] = []
    for r in data.get("regions", []):
        regions.append(
            Region(
                id=str(r["id"]),
                type=str(r.get("type", "other")),
                page=int(r.get("page", 1)),
                bbox_norm=tuple(r["bbox_norm"]),
            )
        )
    return regions


def crop(img: Image.Image, bbox_norm: tuple[float, float, float, float], pad: int = 6) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = bbox_norm
    left = max(int(x0 * w) - pad, 0)
    top = max(int(y0 * h) - pad, 0)
    right = min(int(x1 * w) + pad, w)
    bottom = min(int(y1 * h) + pad, h)
    if right <= left or bottom <= top:
        return img
    return img.crop((left, top, right, bottom))


def main() -> None:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    page_img_path = Path(os.environ.get("PAGE_IMG", "out/page1.png"))
    layout_path = Path(os.environ.get("LAYOUT_JSON", "../rag-ba/layout/page1_layout.json"))
    out_dir = Path(os.environ.get("OUT_DIR", "out_layout_ocr"))
    out_dir.mkdir(parents=True, exist_ok=True)

    assert page_img_path.exists(), f"Missing page image: {page_img_path}"
    assert layout_path.exists(), f"Missing layout json: {layout_path}"

    img = Image.open(page_img_path).convert("RGB")
    regions = load_regions(layout_path)

    # OCR
    ocr = PaddleOCR(lang="en", use_textline_orientation=True)

    outputs = []
    for reg in regions:
        # For MVP: skip sidebar (если layout будет так помечать)
        if reg.type.lower() == "sidebar":
            continue

        patch = crop(img, reg.bbox_norm, pad=8)
        patch_path = out_dir / f"page{reg.page:03d}_{reg.type}_{reg.id}.png"
        patch.save(patch_path)

        result = ocr.predict(str(patch_path))
        lines = deep_get_text_items(result)

        outputs.append(
            {
                "region_id": reg.id,
                "type": reg.type,
                "page": reg.page,
                "bbox_norm": list(reg.bbox_norm),
                "patch_path": str(patch_path),
                "lines": lines,
            }
        )

    out_json = out_dir / "ocr_by_regions.json"
    out_json.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_json} | regions={len(outputs)}")


if __name__ == "__main__":
    main()
