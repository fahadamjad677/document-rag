
from pydantic import BaseModel, ConfigDict, Field

from app.schemas import PageText


class TextChunk(BaseModel):
    """A chunk of text with its original page location."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)
    text: str


def chunk_pages(
    pages: list[PageText],
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[TextChunk]:
    """
    Split page text into fixed-size character windows.

    Pages are processed independently to preserve page-level
    source metadata. Character offsets refer to the original
    PageText.text string.

    Args:
        pages: Extracted PDF pages.
        chunk_size: Maximum characters in each chunk.
        overlap: Characters shared by adjacent chunks on a page.

    Returns:
        Chunks ordered by page and then by position.

    Raises:
        ValueError: If the chunking configuration is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[TextChunk] = []
    stride = chunk_size - overlap

    for page in pages:
        text = page.text
        text_length = len(text)

        if not text.strip():
            continue

        start = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # Remove only whitespace at the window boundaries.
            # Keep offsets aligned with the original page text.
            actual_start = start
            actual_end = end

            while (
                actual_start < actual_end
                and text[actual_start].isspace()
            ):
                actual_start += 1

            while (
                actual_end > actual_start
                and text[actual_end - 1].isspace()
            ):
                actual_end -= 1

            if actual_start < actual_end:
                chunks.append(
                    TextChunk(
                        chunk_index=len(chunks),
                        page_number=page.page_number,
                        start_char=actual_start,
                        end_char=actual_end,
                        text=text[actual_start:actual_end],
                    )
                )

            start += stride

    return chunks
