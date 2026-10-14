# Simulate the output of your PDF extraction function.
from uuid import uuid4

from app.schemas import PDFIngestionResponse, PageText


# extracted_pages = [
#     {
#         "page_number": 1,
#         "text": "Employees receive annual leave according to company policy.",
#     },
#     {
#         "page_number": 2,
#         "text": "Employees must submit leave requests to their manager.",
#     },
# ]

extracted_pages = [
    {"page_number": 1, "text": ""},
    {"page_number": 2, "text": ""},
]

# Validate each extracted page.
pages = [
    PageText.model_validate(page)
    for page in extracted_pages
]

# Calculate the total number of extracted characters.
extracted_characters = sum(
    len(page.text) for page in pages
)

# Build the response.
response = PDFIngestionResponse(
    document_id=str(uuid4()),
    filename="company_policy.pdf",
    page_count=len(pages),
    extracted_characters=extracted_characters,
    extraction_status=(
        "text_extracted"
        if extracted_characters > 0
        else "no_extractable_text"
    ),
    pages=pages,
)

# Print the response as formatted JSON.
print(response.model_dump_json(indent=2))
