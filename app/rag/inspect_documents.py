from pathlib import Path

from app.rag.document_loader import extract_text_from_pdf


DOCUMENTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "documents"
)


def main():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF files.\n")

    for pdf_path in pdf_files:
        try:
            pages = extract_text_from_pdf(str(pdf_path))

            total_characters = sum(
                len(page["text"]) for page in pages
            )

            print(f"Document: {pdf_path.name}")
            print(f"Pages with text: {len(pages)}")
            print(f"Characters extracted: {total_characters}")
            print("-" * 60)

        except Exception as error:
            print(f"ERROR: {pdf_path.name}")
            print(error)
            print("-" * 60)


if __name__ == "__main__":
    main()