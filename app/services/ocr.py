
from io import BytesIO

import pymupdf


OCR_DPI = 150


def extract_page_text_with_ocr(
    page: pymupdf.Page,
) -> str:
    """
    Render a PDF page as an image and extract its text using Tesseract.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are missing. Install pytesseract "
            "and Pillow, and install the Tesseract OCR application."
        ) from exc

    scale = OCR_DPI / 72

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )

    image_bytes = pixmap.tobytes("png")

    with Image.open(BytesIO(image_bytes)) as image:
        text = pytesseract.image_to_string(
            image,
            config="--psm 3",
        )

    return text.strip()
