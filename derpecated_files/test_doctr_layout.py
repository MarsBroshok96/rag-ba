from __future__ import annotations

from pathlib import Path

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.utils.visualization import visualize_page


def main() -> None:
    pdf_path = Path("data/raw/test.pdf")
    if not pdf_path.exists():
        print("Place test.pdf in data/raw/")
        return

    # Load PDF (DocTR renders pages internally)
    doc = DocumentFile.from_pdf(str(pdf_path))

    # OCR predictor = detection + recognition
    # For layout segmentation, detection is the key; recognition helps with text extraction.
    predictor = ocr_predictor(
        det_arch="db_resnet50",
        reco_arch="crnn_vgg16_bn",
        pretrained=True,
    )

    result = predictor(doc)

    # Basic stats
    num_pages = len(result.pages)
    print(f"\nPages: {num_pages}")

    for i, page in enumerate(result.pages, start=1):
        n_blocks = len(page.blocks)
        n_lines = sum(len(b.lines) for b in page.blocks)
        n_words = sum(len(l.words) for b in page.blocks for l in b.lines)
        print(f"Page {i}: blocks={n_blocks}, lines={n_lines}, words={n_words}")

    # Visualize first page with detected boxes
    fig = visualize_page(result.pages[0].export(), doc[0], interactive=False)

    out = Path("doctr_debug_page1.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved visualization to {out.resolve()}")


if __name__ == "__main__":
    main()
