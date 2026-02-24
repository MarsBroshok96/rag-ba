from pathlib import Path

from derpecated_files.pdf_loader import load_pdf_as_documents


def main():
    pdf_path = Path("data/raw/test.pdf")

    if not pdf_path.exists():
        print("Place a test PDF at data/raw/test.pdf")
        return

    docs = load_pdf_as_documents(pdf_path)

    print(f"Loaded {len(docs)} paragraph-documents\n")

    for i, doc in enumerate(docs[:5], start=1):
        print(f"--- Document {i} ---")
        print("Metadata:", doc.metadata)
        print("Text preview:", doc.text[:200])
        print()


if __name__ == "__main__":
    main()
