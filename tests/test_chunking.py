from app.rag.chunking import clean_text, chunk_pages


def test_clean_text():
    text = "Hello     world\n\nthis   is   a test."

    cleaned = clean_text(text)

    assert cleaned == "Hello world this is a test."


def test_chunking():
    pages = [
        {
            "document": "test.pdf",
            "page": 1,
            "text": "A" * 2000,
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=800,
        overlap=150,
    )

    assert len(chunks) > 1
    assert chunks[0]["document"] == "test.pdf"
    assert chunks[0]["page"] == 1
    assert chunks[0]["chunk_id"] == 0
    assert len(chunks[0]["text"]) <= 800


def test_invalid_overlap():
    pages = [
        {
            "document": "test.pdf",
            "page": 1,
            "text": "some text",
        }
    ]

    try:
        chunk_pages(
            pages,
            chunk_size=800,
            overlap=800,
        )
        assert False
    except ValueError:
        assert True