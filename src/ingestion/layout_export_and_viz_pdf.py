from __future__ import annotations

import json
from pathlib import Path

import layoutparser as lp
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

# ================= CONFIG =================

INBOX_DIR = Path("data/inbox/pdf")
OUT_ROOT = Path("layout")

COLOR_MAP = {
    "text": (0, 128, 255),
    "title": (0, 200, 0),
    "list": (255, 165, 0),
    "table": (255, 0, 0),
    "figure": (160, 32, 240),
    "other": (128, 128, 128),
}
DEFAULT_COLOR = (128, 128, 128)

PRIORITY = {"table": 0, "figure": 1, "title": 2, "text": 3, "list": 4, "other": 5}


# ================= UTIL =================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def load_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        return ImageFont.load_default()


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
    sx0, sy0, sx1, sy1 = small
    bx0, by0, bx1, by1 = big
    ix0, iy0 = max(sx0, bx0), max(sy0, by0)
    ix1, iy1 = min(sx1, bx1), min(sy1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    sa = _bbox_area(small)
    return inter / sa if sa > 0 else 0.0


def suppress_cross_class(regions: list[dict], iou_thr=0.35, cover_thr=0.70) -> list[dict]:
    regs = sorted(
        regions,
        key=lambda r: (PRIORITY.get(r.get("type", "other"), 99), -float(r.get("score", 0.0)))
    )

    kept: list[dict] = []
    for r in regs:
        drop = False
        for k in kept:
            if PRIORITY.get(k["type"], 99) <= PRIORITY.get(r["type"], 99):
                if _iou(r["bbox_px"], k["bbox_px"]) >= iou_thr or \
                   _coverage(r["bbox_px"], k["bbox_px"]) >= cover_thr:
                    drop = True
                    break
        if not drop:
            kept.append(r)

    return kept



def suppress_within_type(
    regions: list[dict],
    typ: str,
    *,
    iou_thr: float = 0.50,
    cover_thr: float = 0.80,
) -> list[dict]:
    """
    NMS внутри одного класса: убираем дубликаты (например text поверх text),
    оставляя более "сильный" (по score, затем по площади).
    """
    items = [r for r in regions if r.get("type") == typ and r.get("bbox_px")]
    others = [r for r in regions if r.get("type") != typ or not r.get("bbox_px")]

    items = sorted(
        items,
        key=lambda r: (float(r.get("score", 0.0)), _bbox_area(r["bbox_px"])),
        reverse=True,
    )

    kept: list[dict] = []
    for r in items:
        rb = r["bbox_px"]
        drop = False
        for k in kept:
            kb = k["bbox_px"]
            if _iou(rb, kb) >= iou_thr or _coverage(rb, kb) >= cover_thr:
                drop = True
                break
        if not drop:
            kept.append(r)

    return others + kept


def reclassify_small_titles_as_text(
    regions: list[dict],
    page_w: int,
    *,
    height_factor: float = 1.20,
    min_text_samples: int = 6,
    max_title_width_frac_for_reclass: float = 0.55,
) -> list[dict]:
    """
    Ложные title часто имеют высоту как у обычной строки и небольшую ширину.
    Настоящие заголовки чаще шире (и/или выше), поэтому мы их сохраняем.
    """
    text_heights = [
        (r["bbox_px"][3] - r["bbox_px"][1])
        for r in regions
        if r.get("type") == "text" and r.get("bbox_px") and (r["bbox_px"][3] > r["bbox_px"][1])
    ]
    if len(text_heights) < min_text_samples:
        return regions

    text_heights_sorted = sorted(text_heights)
    median_h = text_heights_sorted[len(text_heights_sorted) // 2]

    for r in regions:
        if r.get("type") != "title":
            continue
        x0, y0, x1, y1 = r["bbox_px"]
        h = max(1, y1 - y0)
        w = max(1, x1 - x0)
        w_frac = w / max(1, page_w)

        # reclassify только если это "строчный" и не широкий заголовок
        if h <= int(median_h * height_factor) and w_frac <= max_title_width_frac_for_reclass:
            r["type"] = "text"

    return regions



def normalize_types(regions: list[dict]) -> list[dict]:
    # list сейчас считаем обычным текстом, чтобы не плодить пересечения и дубликаты
    for r in regions:
        if r.get("type") == "list":
            r["type"] = "text"
    return regions


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
    max_h_gap_px: int = 140,
    fig_iou_drop: float = 0.10,
    fig_cover_drop: float = 0.60,
    text_cover_drop: float = 0.80,
) -> list[dict]:
    """
    Частный хак под PubLayNet:
    Иногда "table" покрывает 70-90% таблицы, а правая колонка/часть таблицы размечается как "figure".
    Тогда:
      - расширяем bbox таблицы до union(table, figure)
      - удаляем соответствующий figure
      - удаляем text/title, которые почти полностью лежат внутри итоговой таблицы

    Важно: эта функция НЕ пытается найти все таблицы, а работает с "главной" таблицей страницы (самая большая).
    """
    tables = [r for r in regions if r.get("type") == "table" and r.get("bbox_px")]
    if not tables:
        return regions

    page_area = max(1, w * h)

    def _key(r):
        return (_bbox_area(r["bbox_px"]), float(r.get("score", 0.0)))

    main_table = sorted(tables, key=_key, reverse=True)[0]
    tb = main_table["bbox_px"]
    tb_area_frac = _bbox_area(tb) / page_area

    # слишком маленькая таблица — не трогаем
    if tb_area_frac < min_table_area_frac:
        return regions

    merged_any = False

    # кандидаты "правой части" — только figure
    for fig in [r for r in regions if r.get("type") == "figure" and r.get("bbox_px")]:
        fb = fig["bbox_px"]

        # вертикальное перекрытие должно быть большим (т.е. фигура по высоте совпадает с таблицей)
        v_ov = _overlap_1d(tb[1], tb[3], fb[1], fb[3])
        if v_ov < min_v_overlap:
            continue

        # горизонтальный гэп между ними (если пересекаются — 0)
        if fb[0] >= tb[2]:
            gap = fb[0] - tb[2]
        elif tb[0] >= fb[2]:
            gap = tb[0] - fb[2]
        else:
            gap = 0

        if gap > max_h_gap_px:
            continue

        # если дошли сюда — считаем, что это “продолжение таблицы”
        tb = _bbox_union(tb, fb)
        merged_any = True

    if not merged_any:
        return regions

    # обновляем таблицу (и bbox_norm тоже)
    main_table["bbox_px"] = tb
    main_table["bbox_norm"] = [
        clamp01(tb[0] / w),
        clamp01(tb[1] / h),
        clamp01(tb[2] / w),
        clamp01(tb[3] / h),
    ]

    # фильтрация регионов после merge
    out: list[dict] = []
    for r in regions:
        rt = r.get("type", "other")
        rb = r.get("bbox_px")

        if not rb:
            out.append(r)
            continue

        # удаляем figure, которые теперь внутри/перекрываются с новой таблицей
        if rt == "figure":
            if _iou(rb, tb) >= fig_iou_drop or _coverage(rb, tb) >= fig_cover_drop:
                continue

        # удаляем текст/тайтлы, которые почти целиком внутри таблицы
        if rt in ("text", "title"):
            if _coverage(rb, tb) >= text_cover_drop:
                continue

        out.append(r)

    return out


# ================= MAIN =================

def main() -> None:

    assert INBOX_DIR.exists(), f"Missing inbox: {INBOX_DIR}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    model = lp.models.Detectron2LayoutModel(
        config_path="models/d2_configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        model_path="models/publaynet_frcnn_model_final.pth",
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure", 5: "other"},
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3,
            "MODEL.ROI_HEADS.NUM_CLASSES", 6,
        ],
    )

    font = load_font()

    pdf_files = sorted(INBOX_DIR.glob("*.pdf"))
    assert pdf_files, "No PDFs found in inbox"

    for pdf_path in pdf_files:

        doc_id = pdf_path.stem
        doc_out = OUT_ROOT / doc_id
        doc_out.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {pdf_path.name}")

        pages = convert_from_path(str(pdf_path), dpi=200)

        for idx, pil_img in enumerate(pages, start=1):

            pil_img = pil_img.convert("RGB")
            w, h = pil_img.size

            page_png = doc_out / f"page{idx:03d}.png"
            pil_img.save(page_png)

            layout = model.detect(pil_img)

            regions = []
            for i, block in enumerate(layout):
                x0, y0, x1, y1 = block.coordinates
                x0, x1 = float(min(x0, x1)), float(max(x0, x1))
                y0, y1 = float(min(y0, y1)), float(max(y0, y1))

                left, top, right, bottom = int(x0), int(y0), int(x1), int(y1)

                regions.append(
                    {
                        "id": f"{doc_id}_p{idx:03d}_r{i+1}",
                        "doc_id": doc_id,
                        "source_pdf": str(pdf_path.resolve()),
                        "type": str(block.type),
                        "page": idx,
                        "bbox_px": [left, top, right, bottom],
                        "bbox_norm": [
                            clamp01(x0 / w),
                            clamp01(y0 / h),
                            clamp01(x1 / w),
                            clamp01(y1 / h),
                        ],
                        "score": float(getattr(block, "score", 0.0)),
                        "page_image_path": str(page_png.resolve()),
                    }
                )

            # 1) list -> text (убираем “разрезание” текста и overlap list/text)
            regions = normalize_types(regions)

            # 2) ложные title -> text (по высоте строки)
            regions = reclassify_small_titles_as_text(regions, page_w=w, height_factor=1.20)
            
            # 3) NMS внутри классов (убираем text-text дубли)
            regions = suppress_within_type(regions, "text", iou_thr=0.50, cover_thr=0.80)
            regions = suppress_within_type(regions, "title", iou_thr=0.50, cover_thr=0.80)

            # 3) убираем наслоения по приоритетам (table/figure вытесняют, text вытесняет title и т.п.)
            regions = suppress_cross_class(regions, iou_thr=0.35, cover_thr=0.70)

            # 4) частный хак: таблица + правые “колонки-figure” (как у тебя)
            regions = merge_table_with_adjacent_figures(regions, w, h)
            
            layout_json = doc_out / f"page{idx:03d}_layout.json"
            layout_json.write_text(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "source_pdf": str(pdf_path.resolve()),
                        "page": idx,
                        "regions": regions,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            # Visualization
            viz = pil_img.copy()
            draw = ImageDraw.Draw(viz)

            for r in regions:
                left, top, right, bottom = r["bbox_px"]
                typ = r["type"]
                score = r["score"]
                color = COLOR_MAP.get(typ, DEFAULT_COLOR)
                label = f"{typ} {score:.2f}"

                draw.rectangle([left, top, right, bottom], outline=color, width=4)

                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]

                draw.rectangle(
                    [left, max(0, top - th - 6), left + tw + 8, top],
                    fill=color,
                )
                draw.text(
                    (left + 4, max(0, top - th - 4)),
                    label,
                    fill="white",
                    font=font,
                )

            viz_path = doc_out / f"page{idx:03d}_layout_viz.png"
            viz.save(viz_path)

            print(f"  Page {idx:02d}: regions={len(regions)}")

    print("\nAll documents processed.")


if __name__ == "__main__":
    main()
