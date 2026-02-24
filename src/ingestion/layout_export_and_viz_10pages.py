from __future__ import annotations

import json
from pathlib import Path

import layoutparser as lp
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

COLOR_MAP = {
    "text": (0, 128, 255),        # blue
    "title": (0, 200, 0),         # green
    "list": (255, 165, 0),        # orange
    "table": (255, 0, 0),         # red
    "figure": (160, 32, 240),     # purple
    "other": (128, 128, 128),     # gray
}
DEFAULT_COLOR = (128, 128, 128)

# ---- Overlap suppression (cross-class) ----

PRIORITY = {"table": 0, "figure": 1, "text": 2, "title": 3, "list": 4, "other": 5}

def _bbox_area(b):
    x0, y0, x1, y1 = b
    return max(0, x1 - x0) * max(0, y1 - y0)

def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = _bbox_area(a) + _bbox_area(b) - inter
    return inter / ua if ua > 0 else 0.0

def _coverage(small, big) -> float:
    # доля small, покрытая big
    sx0, sy0, sx1, sy1 = small
    bx0, by0, bx1, by1 = big
    ix0, iy0 = max(sx0, bx0), max(sy0, by0)
    ix1, iy1 = min(sx1, bx1), min(sy1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    sa = _bbox_area(small)
    return inter / sa if sa > 0 else 0.0

def suppress_cross_class(regions: list[dict], iou_thr=0.35, cover_thr=0.70) -> list[dict]:
    """
    Убираем наслоения между классами:
    - table "выталкивает" figure/text/title/list при сильном перекрытии
    - figure "выталкивает" text внутри графиков
    Требует regions[*]["bbox_px"] = [x0,y0,x1,y1] в пикселях.
    """
    regs = sorted(
        regions,
        key=lambda r: (PRIORITY.get(r.get("type", "other"), 99), -float(r.get("score", 0.0)))
    )

    kept: list[dict] = []
    for r in regs:
        r_type = r.get("type", "other")
        r_box = r["bbox_px"]

        drop = False
        for k in kept:
            k_type = k.get("type", "other")
            k_box = k["bbox_px"]

            # k более приоритетный или равный
            if PRIORITY.get(k_type, 99) <= PRIORITY.get(r_type, 99):
                if _iou(r_box, k_box) >= iou_thr or _coverage(r_box, k_box) >= cover_thr:
                    drop = True
                    break

        if not drop:
            kept.append(r)

    return kept


def _overlap_1d(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    denom = max(1, min(a1 - a0, b1 - b0))
    return inter / denom

def _bbox_union(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return [min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)]

def merge_table_with_adjacent_figures(
    regions: list[dict],
    w: int,
    h: int,
    *,
    min_table_area_frac: float = 0.20,
    min_v_overlap: float = 0.80,
    max_h_gap_px: int = 120,
) -> list[dict]:
    """
    Если таблица большая, а "figure" рядом является по сути правыми колонками таблицы:
    - расширяем bbox таблицы до union(table, figure)
    - удаляем соответствующий figure
    - удаляем text/title/list полностью покрытые итоговой таблицей
    """
    tables = [r for r in regions if r.get("type") == "table"]
    if not tables:
        return regions

    page_area = max(1, w * h)

    # Берём самую "главную" таблицу (по площади, если равны — по score)
    def _key(r):
        a = _bbox_area(r["bbox_px"])
        return (a, float(r.get("score", 0.0)))

    main_table = sorted(tables, key=_key, reverse=True)[0]
    tb = main_table["bbox_px"]
    tb_area_frac = _bbox_area(tb) / page_area

    if tb_area_frac < min_table_area_frac:
        return regions  # слишком маленькая таблица — не трогаем

    # Ищем figure-кандидатов на "правую часть таблицы"
    merged = False
    for fig in [r for r in regions if r.get("type") == "figure"]:
        fb = fig["bbox_px"]

        v_ov = _overlap_1d(tb[1], tb[3], fb[1], fb[3])  # по Y
        if v_ov < min_v_overlap:
            continue

        # Горизонтальный gap между bbox (если пересекаются — gap=0)
        gap = 0
        if fb[0] > tb[2]:
            gap = fb[0] - tb[2]
        elif tb[0] > fb[2]:
            gap = tb[0] - fb[2]

        if gap > max_h_gap_px:
            continue

        # Если figure сильно "смотрится" как продолжение таблицы — объединяем
        tb = _bbox_union(tb, fb)
        merged = True

    if not merged:
        return regions

    # Обновляем main_table bbox
    main_table["bbox_px"] = tb
    main_table["bbox_norm"] = [clamp01(tb[0] / w), clamp01(tb[1] / h), clamp01(tb[2] / w), clamp01(tb[3] / h)]

    # Фильтрация регионов:
    out = []
    for r in regions:
        rt = r.get("type", "other")
        rb = r["bbox_px"]

        # Удаляем figure, которые теперь внутри/рядом и были слиты (критерий: сильное перекрытие с новой таблицей)
        if rt == "figure":
            if _iou(rb, tb) >= 0.10 or _coverage(rb, tb) >= 0.60:
                continue

        # Удаляем текстовые блоки, которые почти целиком внутри итоговой таблицы
        if rt in ("text", "title", "list"):
            if _coverage(rb, tb) >= 0.80:
                continue

        out.append(r)

    return out



def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def load_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        return ImageFont.load_default()


def main() -> None:
    pdf_path = Path("data/inbox/pdf/test.pdf")
    out_dir = Path("layout")
    out_dir.mkdir(parents=True, exist_ok=True)
    assert pdf_path.exists(), f"Missing PDF: {pdf_path}"

    # Model: local detectron2 configs + local PubLayNet weights
    model = lp.Detectron2LayoutModel(
        config_path="models/d2_configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        model_path="models/publaynet_frcnn_model_final.pth",
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure", 5: "other"},
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3,
            "MODEL.ROI_HEADS.NUM_CLASSES", 6,
        ],
    )

    font = load_font()

    # Rasterize 1..10
    pages = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=10)

    for idx, pil_img in enumerate(pages, start=1):
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size

        page_png = out_dir / f"page{idx:03d}.png"
        pil_img.save(page_png)

        layout = model.detect(pil_img)

        regions = []
        for i, block in enumerate(layout):
            x0, y0, x1, y1 = block.coordinates

            # detectron2/layoutparser может дать float; нормализуем и фиксируем порядок координат
            x0, x1 = float(min(x0, x1)), float(max(x0, x1))
            y0, y1 = float(min(y0, y1)), float(max(y0, y1))

            left, top, right, bottom = int(x0), int(y0), int(x1), int(y1)

            regions.append(
                {
                    "id": f"p{idx:03d}_r{i+1}",
                    "type": str(block.type),
                    "page": idx,
                    "bbox_px": [left, top, right, bottom],
                    "bbox_norm": [clamp01(x0 / w), clamp01(y0 / h), clamp01(x1 / w), clamp01(y1 / h)],
                    "score": float(getattr(block, "score", 0.0)),
                }
            )

        # --- NEW: убираем наслоения разных типов ---
        regions = suppress_cross_class(regions, iou_thr=0.35, cover_thr=0.70)
        regions = merge_table_with_adjacent_figures(regions, w, h)


        layout_json = out_dir / f"page{idx:03d}_layout.json"
        layout_json.write_text(json.dumps({"page": idx, "regions": regions}, ensure_ascii=False, indent=2), encoding="utf-8")

        # Visualization
        viz = pil_img.copy()
        draw = ImageDraw.Draw(viz)

        for r in regions:
            left, top, right, bottom = r["bbox_px"]

            typ = r.get("type", "unknown")
            score = r.get("score", 0.0)
            color = COLOR_MAP.get(typ, DEFAULT_COLOR)
            label = f"{typ} {score:.2f}"

            draw.rectangle([left, top, right, bottom], outline=color, width=4)

            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            draw.rectangle([left, max(0, top - th - 6), left + tw + 8, top], fill=color)
            draw.text((left + 4, max(0, top - th - 4)), label, fill="white", font=font)

        viz_path = out_dir / f"page{idx:03d}_layout_viz.png"
        viz.save(viz_path)

        print(f"Page {idx:02d}: regions={len(regions)} saved={layout_json.name} viz={viz_path.name}")

    print("Done. Output in ./layout/")


if __name__ == "__main__":
    main()
