
import pytest

from app.schemas import PDFIngestionResponse, PageText
from app.services.text_chunker import chunk_document


def make_document(
    page_texts: list[str],
) -> PDFIngestionResponse:
    pages = [
        PageText(page_number=index, text=text)
        for index, text in enumerate(page_texts, start=1)
    ]

    return PDFIngestionResponse(
        document_id="doc_test",
        filename="test.pdf",
        page_count=len(pages),
        extracted_characters=sum(
            len(page.text) for page in pages
        ),
        extraction_status=(
            "text_extracted"
            if any(page.text.strip() for page in pages)
            else "no_extractable_text"
        ),
        pages=pages,
    )


def test_short_text_produces_one_chunk() -> None:
    document = make_document(["Hello RAG"])

    chunks = chunk_document(
        document,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Hello RAG"
    assert chunks[0].page_number == 1
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 9


def test_chunks_have_expected_overlap() -> None:
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    document = make_document([text])

    chunks = chunk_document(
        document,
        chunk_size=10,
        overlap=3,
    )

    assert len(chunks) == 4
    assert chunks[0].text == text[0:10]
    assert chunks[1].text == text[7:17]
    assert chunks[2].text == text[14:24]
    assert chunks[3].text == text[21:26]


def test_empty_pages_produce_no_chunks() -> None:
    document = make_document(["", "   ", "\n"])

    chunks = chunk_document(document)

    assert chunks == []


def test_page_metadata_is_preserved() -> None:
    document = make_document([
        "First page content",
        "Second page content",
    ])

    chunks = chunk_document(
        document,
        chunk_size=100,
        overlap=10,
    )

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert all(
        chunk.document_id == "doc_test"
        for chunk in chunks
    )


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_configuration_is_rejected(
    chunk_size: int,
    overlap: int,
) -> None:
    document = make_document(["Some text"])

    with pytest.raises(ValueError):
        chunk_document(
            document,
            chunk_size=chunk_size,
            overlap=overlap,
        )
