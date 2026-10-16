
from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app.schemas import PDFIngestionResponse
from app.services.pdf_extractor import (
    MAX_PDF_SIZE_BYTES,
    PDFValidationError,
    extract_pdf,
)


app = FastAPI(
    title="Document RAG API",
    description="PDF ingestion API for a local RAG application.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/v1/documents/ingest",
    response_model=PDFIngestionResponse,
    status_code=status.HTTP_200_OK,
)
async def ingest_document(
    file: UploadFile = File(...),
) -> PDFIngestionResponse:
    filename = file.filename or "uploaded.pdf"

    try:
        # Read at most 1 byte beyond the limit so we can
        # reject oversized uploads without reading them fully.
        content = await file.read(MAX_PDF_SIZE_BYTES + 1)

        if len(content) > MAX_PDF_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="The PDF exceeds the 10 MB upload limit.",
            )

        try:
            return extract_pdf(filename, content)
        except PDFValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    finally:
        await file.close()
