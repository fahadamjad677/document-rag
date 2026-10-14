
from uuid import uuid4

import pymupdf

from app.services.ocr import extract_page_text_with_ocr
from app.schemas import PDFIngestionResponse, PageText


MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 500


class PDFValidationError(Exception):
    """Raised when a PDF cannot be safely processed."""


def extract_pdf(
    filename: str,
    content: bytes,
    *,
    enable_ocr: bool = False,
) -> PDFIngestionResponse:
    if not content:
        raise PDFValidationError(
            "The uploaded file is empty."
        )

    if len(content) > MAX_PDF_SIZE_BYTES:
        raise PDFValidationError(
            "The PDF exceeds the 10 MiB upload limit."
        )

    if not content.startswith(b"%PDF-"):
        raise PDFValidationError(
            "The uploaded file does not have a valid PDF header."
        )

    try:
        with pymupdf.open(
            stream=content,
            filetype="pdf",
        ) as pdf:
            if pdf.needs_pass:
                raise PDFValidationError(
                    "Password-protected PDFs are not supported."
                )

            page_count = pdf.page_count

            if page_count == 0:
                raise PDFValidationError(
                    "The PDF contains no pages."
                )

            if page_count > MAX_PDF_PAGES:
                raise PDFValidationError(
                    "The PDF exceeds the 500-page limit."
                )

            document_id = str(uuid4())
            pages: list[PageText] = []

            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()
                extraction_method = "text"

                if not text and enable_ocr:
                    text = extract_page_text_with_ocr(page)
                    extraction_method = (
                        "ocr" if text else "none"
                    )

                elif not text:
                    extraction_method = "none"

                pages.append(
                    PageText(
                        page_number=page_index + 1,
                        text=text,
                        extraction_method=extraction_method,
                    )
                )

    except PDFValidationError:
        raise

    except (pymupdf.FileDataError, ValueError, RuntimeError) as exc:
        raise PDFValidationError(
            "The uploaded file is not a readable PDF."
        ) from exc

    extracted_characters = sum(
        len(page.text) for page in pages
    )

    extraction_status = (
        "text_extracted"
        if extracted_characters > 0
        else "no_extractable_text"
    )

    return PDFIngestionResponse(
        document_id=document_id,
        filename=filename,
        page_count=page_count,
        extracted_characters=extracted_characters,
        extraction_status=extraction_status,
        pages=pages,
    )
