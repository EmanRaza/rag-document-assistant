from pathlib import Path
import json

from app.rag.document_loader import extract_text_from_pdf
from app.rag.chunking import chunk_pages


DOCUMENTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "documents"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        pages = extract_text_from_pdf(str(pdf_path))

        chunks = chunk_pages(
            pages,
            chunk_size=800,
            overlap=150,
        )

        all_chunks.extend(chunks)

        print(f"  Pages: {len(pages)}")
        print(f"  Chunks: {len(chunks)}")

    output_file = OUTPUT_DIR / "chunks.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()