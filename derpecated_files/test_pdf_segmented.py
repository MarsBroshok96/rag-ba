from pathlib import Path
from collections import Counter

from derpecated_files.pdf_layout_segmented import load_pdf_segmented


def main():
    pdf_path = Path("data/raw/test.pdf")

    if not pdf_path.exists():
        print("Place a test PDF at data/raw/test.pdf")
        return

    docs = load_pdf_segmented(pdf_path)

    print(f"\nTotal documents: {len(docs)}\n")

    region_counts = Counter(d.metadata.get("region_type", "unknown") for d in docs)
    print("Region type distribution:")
    for k, v in region_counts.items():
        print(f"  {k}: {v}")
    print()

    for i, d in enumerate(docs[:8], start=1):
        print(f"--- Doc {i} ---")
        print("Metadata:", d.metadata)
        print("Text preview:", d.text[:400])
        print()


if __name__ == "__main__":
    main()
