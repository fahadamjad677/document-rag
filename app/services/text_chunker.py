
from uuid import uuid4

from app.schemas import DocumentChunk, PDFIngestionResponse


def chunk_document(
    document: PDFIngestionResponse,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """
    Split extracted PDF page text into overlapping character chunks.

    Character offsets are relative to the original page text.
    Empty and whitespace-only sections are skipped.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []

    for page in document.pages:
        page_text = page.text

        if not page_text.strip():
            continue

        start = 0

        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            chunk_text = page_text[start:end]

            if chunk_text.strip():
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid4()),
                        document_id=document.document_id,
                        filename=document.filename,
                        chunk_index=len(chunks),
                        page_number=page.page_number,
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                    )
                )

            if end == len(page_text):
                break

            # Move forward while retaining the requested overlap.
            start = end - overlap

    return chunks
