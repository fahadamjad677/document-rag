
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PageText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str
    extraction_method: Literal["text", "ocr", "none"] = "text"


class PDFIngestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    page_count: int = Field(ge=0)
    extracted_characters: int = Field(ge=0)
    extraction_status: Literal[
        "text_extracted",
        "no_extractable_text",
    ]
    pages: list[PageText]



class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    filename: str

    # Position of this chunk in the document's chunk list.
    chunk_index: int = Field(ge=0)

    # Original PDF page containing this chunk.
    page_number: int = Field(ge=1)

    # Extracted content for this chunk.
    text: str = Field(min_length=1)

    # Character offsets within the original page text.
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)

