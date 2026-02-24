from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path

import layoutparser as lp
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file, get_checkpoint_url


def main():
    pdf_path = Path("data/raw/test.pdf")
    if not pdf_path.exists():
        print("Place test.pdf in data/raw/")
        return

    # Render first page to image
    images = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1)
    image = np.array(images[0])

    # ---- Detectron2 config via model_zoo (stable, no lp:// yaml) ----
    # We still want PubLayNet weights. LayoutParser provides a convenience wrapper,
    # but the lp:// config download is flaky in some environments.
    #
    # We'll use a standard Faster R-CNN config and load PubLayNet weights via URL.
    # LayoutParser supports passing a Detectron2 cfg + weights path.
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # Label map compatible with PubLayNet-style categories (we'll validate visually)
    label_map = {
        0: "Text",
        1: "Title",
        2: "List",
        3: "Table",
        4: "Figure",
    }

    model = lp.Detectron2LayoutModel(
        config_path=cfg,
        label_map=label_map,
        enforce_cpu=False,
    )

    layout = model.detect(image)

    print("\nDetected blocks:", len(layout))
    counts = {}
    for b in layout:
        counts[b.type] = counts.get(b.type, 0) + 1
    for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")

    vis = lp.draw_box(image, layout, box_width=3, show_element_type=True)
    out = Path("layout_debug_page1.png")
    cv2.imwrite(str(out), vis[:, :, ::-1])  # RGB->BGR
    print(f"\nSaved visualization to {out.resolve()}")


if __name__ == "__main__":
    main()
