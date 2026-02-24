from pathlib import Path

from derpecated_files.pdf_loader_paragraphs import load_pdf_as_paragraph_documents


def main():
    pdf_path = Path("data/raw/test.pdf")

    if not pdf_path.exists():
        print("Place a test PDF at data/raw/test.pdf")
        return

    docs = load_pdf_as_paragraph_documents(pdf_path)

    print(f"\nLoaded {len(docs)} paragraph-block documents\n")

    for i, doc in enumerate(docs[:50], start=1):
        print(f"--- Paragraph {i} ---")
        print("Metadata:", doc.metadata)
        print("Text preview:", doc.text[:])
        print()


if __name__ == "__main__":
    main()
