from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import re

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter, ImageOps  # <-- ADD ImageOps


# ================= UTIL =================

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def crop_roi(
    pil_img: Image.Image,
    bbox_px: list[int],
    *,
    pad_left: int = 32,
    pad_top: int = 24,
    pad_right: int = 32,
    pad_bottom: int = 14,
) -> Image.Image:
    w, h = pil_img.size
    x0, y0, x1, y1 = bbox_px
    x0 = clamp_int(x0 - pad_left, 0, w)
    y0 = clamp_int(y0 - pad_top, 0, h)
    x1 = clamp_int(x1 + pad_right, 0, w)
    y1 = clamp_int(y1 + pad_bottom, 0, h)
    if x1 <= x0 or y1 <= y0:
        return pil_img.crop((0, 0, 1, 1))
    return pil_img.crop((x0, y0, x1, y1))


def preprocess_for_ocr(
    pil_img: Image.Image,
    *,
    scale: int = 3,
    max_side_limit: int = 3990,  # <--- чуть ниже 4000, чтобы OCR не трогал ресайзом
) -> Image.Image:
    img = pil_img.convert("RGB")
    w, h = img.size

    # эффективный scale, чтобы max(w,h)*scale <= max_side_limit
    max_side = max(w, h)
    if max_side > 0:
        scale_eff = min(scale, max(1, int(max_side_limit // max_side)))
    else:
        scale_eff = 1

    if scale_eff > 1:
        img = img.resize((w * scale_eff, h * scale_eff), Image.Resampling.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.6)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return img, scale_eff


def _fit_max_side(pil_img: Image.Image, max_side_limit: int = 3990) -> Image.Image:
    w, h = pil_img.size
    m = max(w, h)
    if m <= max_side_limit:
        return pil_img
    k = max_side_limit / float(m)
    nw, nh = max(1, int(w * k)), max(1, int(h * k))
    return pil_img.resize((nw, nh), Image.Resampling.LANCZOS)


# ================= OCR =================

def ocr_image(ocr: PaddleOCR, pil_img: Image.Image) -> list[dict[str, Any]]:
    """
    Возвращает список строк (line-level) в координатах ROI.
    Поддерживает 2 формата вывода:
      A) legacy: [ [ [box, (text, score)], ... ] ]
      B) paddleocr>=3: list[dict] (rec_texts/rec_scores/rec_polys/dt_polys)
    """

    def parse_res(res_any, scale_div: float = 1.0) -> list[dict[str, Any]]:
        if not res_any:
            return []

        items = res_any
        if isinstance(items, list) and len(items) == 1:
            if isinstance(items[0], list):
                items = items[0]
            elif isinstance(items[0], dict):
                pass

        out_lines: list[dict[str, Any]] = []

        # --- page dict format: [ { rec_texts, rec_scores, rec_polys/dt_polys } ] ---
        if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
            page = items[0]
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            polys = page.get("rec_polys") or page.get("dt_polys") or []

            n = min(len(texts), len(scores), len(polys))
            for i in range(n):
                poly = np.array(polys[i]).tolist()
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                if scale_div != 1.0:
                    bbox = [int(v / scale_div) for v in bbox]
                out_lines.append({"text": str(texts[i]), "score": float(scores[i]), "bbox_px": bbox})
            return out_lines

        # --- legacy list format ---
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

        # --- list of dict items ---
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

    # 1) normal pass
    pil_norm = _fit_max_side(pil_img, max_side_limit=3990)
    img0 = np.array(pil_norm.convert("RGB"))
    res0 = ocr.predict(img0) if hasattr(ocr, "predict") else ocr.ocr(img0)
#    out0 = parse_res(res0, scale_div=1.0)
    out0 = parse_res(res0, scale_div=(pil_norm.width / pil_img.width))

    # 2) strong pass
    pil_strong, scale_eff = preprocess_for_ocr(pil_img, scale=3, max_side_limit=3990)
    img1 = np.array(pil_strong)
    res1 = ocr.predict(img1) if hasattr(ocr, "predict") else ocr.ocr(img1)
    out1 = parse_res(res1, scale_div=float(scale_eff))

    # 3) pick best
    if not out0:
        return out1
    if avg_score(out1) > avg_score(out0) + 0.05:
        return out1
    return out0


# ================= BILINGUAL PICKER =================

RE_CYR = re.compile(r"[А-Яа-яЁё]")
RE_LAT = re.compile(r"[A-Za-z]")
RE_ALNUM = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")


def _avg_score(lines: list[dict[str, Any]]) -> float:
    if not lines:
        return 0.0
    return sum(float(x.get("score", 0.0)) for x in lines) / len(lines)


def _script_mix_ratio(text: str) -> tuple[float, float]:
    """
    (cyr_ratio, lat_ratio) среди букв (кириллица+латиница).
    Цифры/прочее не учитываем как индикатор языка.
    """
    t = text or ""
    c = len(RE_CYR.findall(t))
    l = len(RE_LAT.findall(t))
    denom = max(1, c + l)
    return (c / denom, l / denom)


def _noise_penalty(text: str) -> float:
    """
    Штраф за "мусорность": если много символов не из [буквы/цифры].
    """
    t = (text or "").replace(" ", "")
    if not t:
        return 0.0
    alnum_cnt = len(RE_ALNUM.findall(t))
    noise = 1.0 - (alnum_cnt / max(1, len(t)))
    return 0.10 * noise  # 0..0.10


def _quality(lines: list[dict[str, Any]], prefer: str) -> float:
    """
    Сквозная метрика выбора EN vs RU:
    - базово: средний score
    - бонус за доминирующую письменность
    - штраф за шум
    """
    if not lines:
        return 0.0

    txt = " ".join((x.get("text") or "") for x in lines)
    avg = _avg_score(lines)
    cyr, lat = _script_mix_ratio(txt)

    bonus = 0.0
    if prefer == "ru":
        bonus += 0.15 * cyr
    elif prefer == "en":
        bonus += 0.15 * lat
    else:
        bonus += 0.10 * max(cyr, lat)

    return avg + bonus - _noise_penalty(txt)


def ocr_image_bilingual(
    ocr_en: PaddleOCR,
    ocr_ru: PaddleOCR,
    pil_img: Image.Image,
) -> list[dict[str, Any]]:
    """
    Делает OCR двумя моделями (EN+RU) и выбирает лучший результат.
    ВАЖНО: обе модели используют одинаковый pipeline (normal+strong) внутри ocr_image().
    """
    out_en = ocr_image(ocr_en, pil_img)
    out_ru = ocr_image(ocr_ru, pil_img)

    q_en = _quality(out_en, prefer="en")
    q_ru = _quality(out_ru, prefer="ru")

    return out_ru if q_ru >= q_en else out_en


# ================= OCR MARGIN FIX (right-edge truncation) =================

def _apply_margin_and_fix_bboxes(
    lines: list[dict[str, Any]],
    *,
    left: int,
    top: int,
    roi_w: int,
    roi_h: int,
) -> list[dict[str, Any]]:
    """
    Мы OCR делали на ROI с белым бордером.
    Возвращаем bbox_px обратно в координаты исходного ROI (без бордера).
    """
    out: list[dict[str, Any]] = []
    for ln in lines:
        bbox = ln.get("bbox_px")
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            x0, y0, x1, y1 = bbox
            x0 -= left
            x1 -= left
            y0 -= top
            y1 -= top

            # clamp обратно в исходный ROI
            x0 = clamp_int(int(x0), 0, roi_w)
            x1 = clamp_int(int(x1), 0, roi_w)
            y0 = clamp_int(int(y0), 0, roi_h)
            y1 = clamp_int(int(y1), 0, roi_h)

            ln = dict(ln)
            ln["bbox_px"] = [x0, y0, x1, y1]
        out.append(ln)
    return out


def ocr_image_bilingual_with_margin(
    ocr_en: PaddleOCR,
    ocr_ru: PaddleOCR,
    pil_img: Image.Image,
    *,
    pad_left: int = 12,
    pad_top: int = 10,
    pad_right: int = 48,   # <-- ключевое: больше справа
    pad_bottom: int = 14,
) -> list[dict[str, Any]]:
    """
    Добавляем белый бордер вокруг ROI перед OCR (особенно справа),
    чтобы не "съедались" последние буквы.
    Потом возвращаем bbox_px обратно в координаты исходного ROI.
    """
    roi_w, roi_h = pil_img.size

    # белый бордер (фон белый, как у PDF)
    padded = ImageOps.expand(
        pil_img,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=(255, 255, 255),
    )

    lines = ocr_image_bilingual(ocr_en, ocr_ru, padded)

    # bbox_px у линий сейчас в системе padded-ROI → сдвигаем обратно
    return _apply_margin_and_fix_bboxes(lines, left=pad_left, top=pad_top, roi_w=roi_w, roi_h=roi_h)


# ================= PIPELINE =================

def _iter_layout_jsons(layout_root: Path) -> list[Path]:
    """
    layout_root ожидается как: ../rag-ba/layout/<doc_id>/
    внутри — pageXXX_layout.json
    """
    return sorted(layout_root.glob("page*_layout.json"))


def main() -> None:
    # --- input from rag-ba ---
    rag_ba_root = (Path(__file__).resolve().parent / ".." / "..").resolve()
    in_layout_root = (rag_ba_root / "layout").resolve()

    assert in_layout_root.exists(), f"Input layout not found: {in_layout_root}"

    # --- output in rag-ba-ocr ---
    out_root = Path("layout_ocr")
    out_root.mkdir(parents=True, exist_ok=True)

    # OCR engines (две языковые модели)
    ocr_en = PaddleOCR(lang="en", use_textline_orientation=True)
    ocr_ru = PaddleOCR(lang="ru", use_textline_orientation=True)

    # docs are subfolders under rag-ba/layout/
    doc_dirs = sorted([p for p in in_layout_root.iterdir() if p.is_dir()])
    assert doc_dirs, f"No doc folders found in {in_layout_root} (expected layout/<doc_id>/...)"

    for doc_dir in doc_dirs:
        doc_id = doc_dir.name
        layout_jsons = _iter_layout_jsons(doc_dir)
        if not layout_jsons:
            print(f"Skip doc {doc_id}: no page*_layout.json found")
            continue

        doc_out = out_root / doc_id
        doc_out.mkdir(parents=True, exist_ok=True)

        print(f"\nDOC: {doc_id} pages={len(layout_jsons)}")

        for layout_json_path in layout_jsons:
            data = json.loads(layout_json_path.read_text(encoding="utf-8"))
            page = int(data.get("page", 0))
            regions = data.get("regions", [])

            # page image path: prefer metadata, fallback to sibling pageXXX.png
            page_png = data.get("page_image_path") or data.get("image")
            if not page_png:
                # fallback: in rag-ba layout folder page images are next to json
                guess = layout_json_path.with_name(f"page{page:03d}.png")
                page_png = str(guess)

            page_png_path = Path(page_png)
            if not page_png_path.is_absolute():
                # relative paths are relative to rag-ba root typically
                page_png_path = (rag_ba_root / page_png_path).resolve()

            if not page_png_path.exists():
                print(f"  Skip page {page:03d}: missing page image {page_png_path}")
                continue

            pil_page = Image.open(page_png_path).convert("RGB")
            W, H = pil_page.size

            page_out: dict[str, Any] = {
                "doc_id": data.get("doc_id", doc_id),
                "source_pdf": data.get("source_pdf"),
                "page": page,
                "image": str(page_png_path),
                "size_px": [W, H],
                "regions": [],
            }

            # crops dir per page
            crops_dir = doc_out / f"page{page:03d}_crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

            for r in regions:
                rid = r.get("id")
                rtype = r.get("type", "other")
                bbox_px = r.get("bbox_px")

                if not rid or not bbox_px:
                    # если вдруг bbox_px нет, пересчитаем из norm
                    if not bbox_px and r.get("bbox_norm"):
                        x0n, y0n, x1n, y1n = r["bbox_norm"]
                        bbox_px = [int(x0n * W), int(y0n * H), int(x1n * W), int(y1n * H)]
                    else:
                        continue

                roi = crop_roi(pil_page, bbox_px)

                roi_path = crops_dir / f"{rid}_{rtype}.png"
                roi.save(roi_path)

                lines: list[dict[str, Any]] = []
                if rtype in ("text", "title", "list", "table"):
                    lines = ocr_image_bilingual_with_margin(
                        ocr_en,
                        ocr_ru,
                        roi,
                        pad_left=12,
                        pad_top=10,
                        pad_right=64,
                        pad_bottom=14,
                    )

                page_out["regions"].append(
                    {
                        **r,
                        "doc_id": data.get("doc_id", doc_id),
                        "source_pdf": data.get("source_pdf"),
                        "page_image_path": str(page_png_path),
                        "crop_path": str(roi_path),
                        "ocr_lines": lines,
                        "ocr_line_count": len(lines),
                    }
                )

            out_json = doc_out / f"page{page:03d}_layout_ocr.json"
            out_json.write_text(json.dumps(page_out, ensure_ascii=False, indent=2), encoding="utf-8")

            total_lines = sum(rr["ocr_line_count"] for rr in page_out["regions"])
            print(f"  Page {page:03d}: regions={len(regions)} ocr_lines={total_lines} saved={out_json.name}")

    print("\nDone. Output in ./layout_ocr/<doc_id>/...")


if __name__ == "__main__":
    main()