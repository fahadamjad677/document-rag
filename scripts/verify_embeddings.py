
import asyncio

from app.services.embedding_service import EmbeddingService


async def main() -> None:
    service = EmbeddingService()

    texts = [
        "Employees receive 20 days of annual leave.",
        "The company provides paid vacation days to employees.",
        "The database stores product inventory records.",
    ]

    embeddings = await service.embed_texts(texts)

    print(f"Texts embedded: {len(embeddings)}")
    print(f"Vector dimensions: {len(embeddings[0])}")

    for index, (text, vector) in enumerate(
        zip(texts, embeddings, strict=True),
        start=1,
    ):
        print(f"\nText {index}: {text}")
        print(f"First 5 vector values: {vector[:5]}")


if __name__ == "__main__":
    asyncio.run(main())
