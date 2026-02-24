from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageEnhance, ImageFilter

from paddleocr import PaddleOCR


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def crop_roi(pil_img: Image.Image, bbox_px: list[int], pad: int = 12, pad_left: int = 32, pad_top: int = 24, pad_right: int = 32, pad_bottom: int = 14) -> Image.Image:

    w, h = pil_img.size
    x0, y0, x1, y1 = bbox_px
    x0 = clamp_int(x0 - pad_left, 0, w)
    y0 = clamp_int(y0 - pad_top, 0, h)
    x1 = clamp_int(x1 + pad_right, 0, w)
    y1 = clamp_int(y1 + pad_bottom, 0, h)
    if x1 <= x0 or y1 <= y0:
        return pil_img.crop((0, 0, 1, 1))
    return pil_img.crop((x0, y0, x1, y1))

def preprocess_for_ocr(pil_img: Image.Image, scale: int = 3) -> Image.Image:
    # 1) upscale
    img = pil_img.convert("RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)

    # 2) contrast + sharpness (мягко, без жесткой бинаризации)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.6)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return img


def ocr_image(ocr: PaddleOCR, pil_img: Image.Image) -> list[dict[str, Any]]:
    """
    Возвращает список строк (line-level) в координатах ROI.
    Поддерживает 2 формата вывода:
      A) legacy: [ [ [box, (text, score)], ... ] ]
      B) paddleocr>=3: list[dict] (rec_texts/rec_scores/rec_polys/dt_polys)
    """

    def parse_res(res_any, scale_div: float = 1.0) -> list[dict[str, Any]]:
        """
        Унифицированный парсер для разных форматов PaddleOCR.
        scale_div: если OCR запускался на upscale изображении (например, scale=3),
                   то bbox_px делим на scale_div, чтобы вернуть координаты в ROI.
        """
        if not res_any:
            return []

        # Частая вложенность: [ {...} ] или [ [...items...] ]
        items = res_any
        if isinstance(items, list) and len(items) == 1:
            if isinstance(items[0], list):
                items = items[0]
            elif isinstance(items[0], dict):
                # оставим как есть, обработаем ниже
                pass

        out_lines: list[dict[str, Any]] = []

        # --- Формат "страница-словарь": [ { rec_texts, rec_scores, rec_polys/dt_polys } ] ---
        if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
            page = items[0]
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            polys = page.get("rec_polys") or page.get("rec_polys") or page.get("dt_polys") or []

            n = min(len(texts), len(scores), len(polys))
            for i in range(n):
                poly = np.array(polys[i]).tolist()  # (4,2)
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                if scale_div != 1.0:
                    bbox = [int(v / scale_div) for v in bbox]
                out_lines.append(
                    {
                        "text": str(texts[i]),
                        "score": float(scores[i]),
                        "bbox_px": bbox,
                    }
                )
            return out_lines

        # --- Формат legacy: [ [box, (text, score)], ... ] ---
        if isinstance(items, list) and items and isinstance(items[0], (list, tuple)) and len(items[0]) >= 2:
            for it in items:
                if not isinstance(it, (list, tuple)) or len(it) < 2:
                    continue
                box = it[0]
                txt_score = it[1]
                if not box or not txt_score:
                    continue
                txt, score = txt_score
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                if scale_div != 1.0:
                    bbox = [int(v / scale_div) for v in bbox]
                out_lines.append({"text": str(txt), "score": float(score), "bbox_px": bbox})
            return out_lines

        # --- Формат "список dict-элементов" (на всякий случай) ---
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                txt = it.get("rec_text") or it.get("text") or ""
                score = it.get("rec_score") or it.get("score") or 0.0
                poly = it.get("rec_poly") or it.get("dt_poly") or it.get("dt_polys") or it.get("points") or it.get("bbox")

                bbox = None
                if poly is not None:
                    try:
                        poly_list = np.array(poly).tolist()
                        if isinstance(poly_list, list) and len(poly_list) >= 4:
                            xs = [p[0] for p in poly_list if isinstance(p, list) and len(p) >= 2]
                            ys = [p[1] for p in poly_list if isinstance(p, list) and len(p) >= 2]
                            if xs and ys:
                                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                                if scale_div != 1.0:
                                    bbox = [int(v / scale_div) for v in bbox]
                    except Exception:
                        bbox = None

                if txt:
                    out_lines.append({"text": str(txt), "score": float(score), "bbox_px": bbox})
            return out_lines

        return []

    def avg_score(lines: list[dict[str, Any]]) -> float:
        if not lines:
            return 0.0
        return sum(float(x.get("score", 0.0)) for x in lines) / len(lines)

    # 1) обычный проход
    img0 = np.array(pil_img.convert("RGB"))
    res0 = ocr.predict(img0) if hasattr(ocr, "predict") else ocr.ocr(img0)
    out0 = parse_res(res0, scale_div=1.0)
    if not out0:
        # не возвращаем сразу: иногда normal пустой, а strong вытаскивает
        pass

    # 2) усиленный проход (upscale+enhance)
    pil_strong = preprocess_for_ocr(pil_img, scale=3)
    img1 = np.array(pil_strong)
    res1 = ocr.predict(img1) if hasattr(ocr, "predict") else ocr.ocr(img1)
    out1 = parse_res(res1, scale_div=3.0)  # делим bbox обратно

    # 3) выбираем лучший
    # - если normal пустой → берем strong
    # - иначе если strong ощутимо лучше по среднему score → берем strong
    if not out0:
        return out1
    if avg_score(out1) > avg_score(out0) + 0.05:
        return out1
    return out0




def main() -> None:
    layout_dir = Path("layout")
    out_dir = Path("layout_ocr")
    out_dir.mkdir(parents=True, exist_ok=True)

    # PaddleOCR: GPU включается автоматически, если paddle compiled with CUDA + CUDA доступен
    ocr = PaddleOCR(
        lang="en",
        use_textline_orientation=True,  # вместо use_angle_cls
    )

    # Обрабатываем page001..page010 
    for page in range(1, 11):
        page_png = layout_dir / f"page{page:03d}.png"
        layout_json = layout_dir / f"page{page:03d}_layout.json"
        if not page_png.exists() or not layout_json.exists():
            print(f"Skip page {page:03d}: missing inputs")
            continue

        pil_page = Image.open(page_png).convert("RGB")
        W, H = pil_page.size
        data = json.loads(layout_json.read_text(encoding="utf-8"))
        regions = data.get("regions", [])

        page_out: dict[str, Any] = {
            "page": page,
            "image": str(page_png),
            "size_px": [W, H],
            "regions": [],
        }

        # Директории под кропы
        crops_dir = out_dir / f"page{page:03d}_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        for r in regions:
            rid = r["id"]
            rtype = r.get("type", "other")
            bbox_px = r.get("bbox_px")
            if not bbox_px:
                # если вдруг bbox_px нет, пересчитаем из norm
                x0n, y0n, x1n, y1n = r["bbox_norm"]
                bbox_px = [int(x0n * W), int(y0n * H), int(x1n * W), int(y1n * H)]

            roi = crop_roi(pil_page, bbox_px)

            # сохраняем кропы для контроля качества
 #           roi_for_ocr = roi.resize((roi.width * 2, roi.height * 2), Image.Resampling.LANCZOS)
            roi_path = crops_dir / f"{rid}_{rtype}.png"
            roi.save(roi_path)

            # OCR только для text/title/list/table (figure пока не OCRим — обычно мусор)
            lines: list[dict[str, Any]] = []
            if rtype in ("text", "title", "list", "table"):
                lines = ocr_image(ocr, roi)

            page_out["regions"].append(
                {
                    **r,
                    "crop_path": str(roi_path),
                    "ocr_lines": lines,
                    "ocr_line_count": len(lines),
                }
            )

        out_json = out_dir / f"page{page:03d}_layout_ocr.json"
        out_json.write_text(json.dumps(page_out, ensure_ascii=False, indent=2), encoding="utf-8")

        total_lines = sum(rr["ocr_line_count"] for rr in page_out["regions"])
        print(f"Page {page:02d}: regions={len(regions)} ocr_lines={total_lines} saved={out_json.name}")

    print("Done. Output in ./layout_ocr/")


if __name__ == "__main__":
    main()
