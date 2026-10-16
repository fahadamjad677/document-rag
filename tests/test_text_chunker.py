
import pytest

from app.schemas import PageText
from app.services.text_chunker import chunk_pages


def make_page(
    page_number: int,
    text: str,
) -> PageText:
    return PageText(
        page_number=page_number,
        text=text,
    )


def test_empty_page_list_returns_no_chunks() -> None:
    assert chunk_pages([]) == []


def test_whitespace_only_pages_are_skipped() -> None:
    pages = [
        make_page(1, ""),
        make_page(2, "   \n\t "),
    ]

    assert chunk_pages(pages) == []


def test_short_page_produces_one_chunk() -> None:
    pages = [make_page(1, "Hello, RAG!")]

    chunks = chunk_pages(
        pages,
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Hello, RAG!"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len("Hello, RAG!")


def test_chunks_respect_size_and_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_pages(
        [make_page(1, text)],
        chunk_size=10,
        overlap=3,
    )

    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]

    assert chunks[0].text[7:] == chunks[1].text[:3]
    assert chunks[1].text[7:] == chunks[2].text[:3]


def test_chunk_offsets_match_original_page_text() -> None:
    text = "  Annual leave policy applies to employees.  "

    chunks = chunk_pages(
        [make_page(4, text)],
        chunk_size=15,
        overlap=5,
    )

    for chunk in chunks:
        assert (
            text[chunk.start_char:chunk.end_char]
            == chunk.text
        )


def test_page_numbers_are_preserved() -> None:
    pages = [
        make_page(2, "First page content."),
        make_page(7, "Second page content."),
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=100,
        overlap=10,
    )

    assert [chunk.page_number for chunk in chunks] == [2, 7]


def test_chunk_indices_are_sequential() -> None:
    pages = [
        make_page(1, "a" * 25),
        make_page(2, "b" * 25),
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=10,
        overlap=2,
    )

    assert [chunk.chunk_index for chunk in chunks] == list(
        range(len(chunks))
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
    with pytest.raises(ValueError):
        chunk_pages(
            [make_page(1, "Example text.")],
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_whitespace_is_trimmed_without_breaking_offsets() -> None:
    text = "  Hello world  "

    chunks = chunk_pages(
        [make_page(1, text)],
        chunk_size=8,
        overlap=2,
    )

    assert chunks[0].text == "Hello"
    assert text[
        chunks[0].start_char:chunks[0].end_char
    ] == chunks[0].text
