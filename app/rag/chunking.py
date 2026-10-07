def clean_text(text: str) -> str:
    """
    Basic text cleaning.

    Removes excessive whitespace while preserving
    the actual content of the document.
    """

    return " ".join(text.split())


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """
    Split page-level text into overlapping chunks.

    Each chunk retains:
    - chunk_id
    - document
    - page
    - text
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    chunk_id = 0

    for page in pages:
        text = clean_text(page["text"])

        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document": page["document"],
                        "page": page["page"],
                        "text": chunk_text,
                    }
                )

                chunk_id += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks