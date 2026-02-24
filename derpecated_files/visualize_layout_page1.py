from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Цвета по типу региона
COLOR_MAP = {
    "text": (0, 128, 255),        # синий
    "title": (0, 200, 0),         # зеленый
    "list": (255, 165, 0),        # оранжевый
    "table": (255, 0, 0),         # красный
    "figure": (160, 32, 240),     # фиолетовый
}

DEFAULT_COLOR = (128, 128, 128)   # серый


def main() -> None:
    img_path = Path("layout/page1.png")
    layout_path = Path("layout/page1_layout.json")
    out_path = Path("layout/page1_layout_viz.png")

    assert img_path.exists(), f"Missing {img_path}"
    assert layout_path.exists(), f"Missing {layout_path}"

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    data = json.loads(layout_path.read_text(encoding="utf-8"))
    regions = data.get("regions", [])

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for r in regions:
        x0, y0, x1, y1 = r["bbox_norm"]
        left = int(x0 * w)
        top = int(y0 * h)
        right = int(x1 * w)
        bottom = int(y1 * h)

        typ = r.get("type", "unknown")
        score = r.get("score", 0.0)

        color = COLOR_MAP.get(typ, DEFAULT_COLOR)
        label = f"{typ} {score:.2f}"

        # Рисуем рамку
        draw.rectangle([left, top, right, bottom], outline=color, width=4)

        # Фон для подписи
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.rectangle(
            [left, max(0, top - th - 6), left + tw + 8, top],
            fill=color
        )
        draw.text(
            (left + 4, max(0, top - th - 4)),
            label,
            fill="white",
            font=font,
        )

    img.save(out_path)
    print(f"Saved: {out_path} | regions={len(regions)}")


if __name__ == "__main__":
    main()
