
import asyncio

from app.schemas import PageText
from app.services.embedding_service import EmbeddingService
from app.services.retrieval import retrieve_top_k
from app.services.text_chunker import chunk_pages


async def main() -> None:
    pages = [
        PageText(
            page_number=1,
            text=(
                "Annual Leave Policy\n"
                "Full-time employees receive 20 days of "
                "annual leave per calendar year. "
                "Unused leave is subject to company policy."
            ),
        ),
        PageText(
            page_number=2,
            text=(
                "Sick Leave Policy\n"
                "Employees may request sick leave when "
                "they are unable to work due to illness. "
                "Supporting documentation may be required."
            ),
        ),
        PageText(
            page_number=3,
            text=(
                "Inventory Management\n"
                "The warehouse tracks product quantities, "
                "stock movements, and supplier deliveries."
            ),
        ),
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=200,
        overlap=30,
    )

    if not chunks:
        print("No chunks were generated.")
        return

    embedding_service = EmbeddingService()

    # Embed each passage using the local embedding model.
    chunk_embeddings = await embedding_service.embed_texts(
        [chunk.text for chunk in chunks]
    )

    question = "How many annual leave days does an employee receive?"

    query_embeddings = await embedding_service.embed_texts(
        [question]
    )

    results = retrieve_top_k(
        query_embeddings[0],
        chunks,
        chunk_embeddings,
        top_k=3,
    )

    print(f"Question: {question}\n")
    print("Retrieved passages:")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Page: {result.chunk.page_number}")
        print(f"Similarity: {result.similarity:.4f}")
        print(f"Passage: {result.chunk.text}")


if __name__ == "__main__":
    asyncio.run(main())
