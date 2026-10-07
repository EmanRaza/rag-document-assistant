from pathlib import Path

from app.rag.document_loader import extract_text_from_pdf


PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "documents"
    / "attention_is_all_you_need.pdf"
)


def test_pdf_text_extraction():
    pages = extract_text_from_pdf(str(PDF_PATH))

    assert len(pages) > 0
    assert pages[0]["document"] == "attention_is_all_you_need.pdf"
    assert pages[0]["page"] == 1
    assert len(pages[0]["text"]) > 0