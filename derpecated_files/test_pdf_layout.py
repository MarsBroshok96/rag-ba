from pathlib import Path

from derpecated_files.pdf_loader_layout import load_pdf_with_layout


def main():
    pdf_path = Path("data/raw/test.pdf")

    if not pdf_path.exists():
        print("Place a test PDF at data/raw/test.pdf")
        return

    docs = load_pdf_with_layout(pdf_path)

    print(f"\nLoaded {len(docs)} layout-block documents\n")

    for i, doc in enumerate(docs[:5], start=1):
        print(f"--- Block {i} ---")
        print("Metadata:", doc.metadata)
        print("Text preview:", doc.text[:200])
        print()


if __name__ == "__main__":
    main()
