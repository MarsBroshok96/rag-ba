from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return to_jsonable(vars(obj))
    return str(obj)


def deep_get_text_items(root: Any) -> list[dict]:
    """
    Robust extractor for PaddleOCR 3.x: walk nested dict/object and collect
    any (text, score, box) triplets when found.
    """
    items: list[dict] = []

    def walk(x: Any) -> None:
        # dict-like
        if isinstance(x, dict):
            # Common patterns: texts/scores/boxes arrays
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

        # list/tuple
        if isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
            return

        # object with __dict__
        if hasattr(x, "__dict__"):
            walk(vars(x))
            return

    walk(root)

    # de-duplicate by (text, score, box) stringified
    seen = set()
    uniq: list[dict] = []
    for it in items:
        key = (it.get("text"), round(float(it.get("score", 0.0)), 6), json.dumps(it.get("box"), ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    return uniq


def main() -> None:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    pdf_path = Path(os.environ.get("PDF_PATH", "data/test.pdf"))
    out_dir = Path(os.environ.get("OUT_DIR", "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"

    # Rasterize page 1 to PNG on disk (important for PaddleOCR 3.x predict() path mode)
    pages = convert_from_path(str(pdf_path), dpi=250, first_page=1, last_page=1)
    img_rgb = pages[0].convert("RGB")
    img_path = out_dir / "page1.png"
    img_rgb.save(img_path)

    ocr = PaddleOCR(
        lang="en",
        use_textline_orientation=True,
    )

    # IMPORTANT: pass file path, not numpy array
    result = ocr.predict(str(img_path))

    (out_dir / "page1_ocr_raw.json").write_text(
        json.dumps(to_jsonable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    items = deep_get_text_items(result)
    (out_dir / "page1_ocr.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {img_path}")
    print(f"Saved: {out_dir/'page1_ocr_raw.json'}")
    print(f"Saved: {out_dir/'page1_ocr.json'} | lines={len(items)}")


if __name__ == "__main__":
    main()
