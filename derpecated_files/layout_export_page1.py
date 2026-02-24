from __future__ import annotations

import json
from pathlib import Path

import layoutparser as lp
from pdf2image import convert_from_path


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> None:
    pdf_path = Path("data/raw/test.pdf")
    out_dir = Path("layout")
    out_dir.mkdir(parents=True, exist_ok=True)

    assert pdf_path.exists(), f"Missing PDF: {pdf_path}"

    # Rasterize first page
    pages = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1)
    img = pages[0].convert("RGB")
    w, h = img.size
    img_path = out_dir / "page1.png"
    img.save(img_path)

    # PubLayNet detector: TEXT / TITLE / LIST / TABLE / FIGURE
    model = lp.Detectron2LayoutModel(
        config_path="models/d2_configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        model_path="models/publaynet_frcnn_model_final.pth",
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure", 5: "other"},
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3,
            "MODEL.ROI_HEADS.NUM_CLASSES", 6,
        ],
    )

    layout = model.detect(img)

    regions = []
    for i, block in enumerate(layout):
        x0, y0, x1, y1 = block.coordinates  # absolute pixels
        regions.append(
            {
                "id": f"r{i+1}",
                "type": str(block.type),
                "page": 1,
                "bbox_norm": [
                    clamp01(x0 / w),
                    clamp01(y0 / h),
                    clamp01(x1 / w),
                    clamp01(y1 / h),
                ],
                "score": float(getattr(block, "score", 0.0)),
            }
        )

    out_json = out_dir / "page1_layout.json"
    out_json.write_text(json.dumps({"page": 1, "regions": regions}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {img_path}")
    print(f"Saved: {out_json} | regions={len(regions)}")


if __name__ == "__main__":
    main()
