
import json

import httpx
import pytest

from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
)


@pytest.mark.asyncio
async def test_embed_texts_returns_vectors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"

        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        service = EmbeddingService(client=client)

        result = await service.embed_texts(
            ["Annual leave policy", "Sick leave policy"]
        )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


@pytest.mark.asyncio
async def test_empty_input_returns_empty_list() -> None:
    service = EmbeddingService()

    assert await service.embed_texts([]) == []


@pytest.mark.asyncio
async def test_blank_text_is_rejected() -> None:
    service = EmbeddingService()

    with pytest.raises(ValueError, match="cannot be empty"):
        await service.embed_texts(["Annual leave", "  "])


@pytest.mark.asyncio
async def test_wrong_number_of_vectors_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2]]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        service = EmbeddingService(client=client)

        with pytest.raises(
            EmbeddingServiceError,
            match="does not match",
        ):
            await service.embed_texts(["First", "Second"])


@pytest.mark.asyncio
async def test_non_numeric_vector_values_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, "invalid"]]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        service = EmbeddingService(client=client)

        with pytest.raises(
            EmbeddingServiceError,
            match="numeric",
        ):
            await service.embed_texts(["Annual leave"])


@pytest.mark.asyncio
async def test_inconsistent_dimensions_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2],
                    [0.3, 0.4, 0.5],
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        service = EmbeddingService(client=client)

        with pytest.raises(
            EmbeddingServiceError,
            match="inconsistent dimensions",
        ):
            await service.embed_texts(["First", "Second"])


@pytest.mark.asyncio
async def test_http_error_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model failure"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        service = EmbeddingService(client=client)

        with pytest.raises(
            EmbeddingServiceError,
            match="HTTP 500",
        ):
            await service.embed_texts(["Annual leave"])



@pytest.mark.asyncio
async def test_non_finite_vector_values_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(
                {"embeddings": [[float("inf")]]},
                allow_nan=True,
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        service = EmbeddingService(client=client)

        with pytest.raises(
            EmbeddingServiceError,
            match="finite",
        ):
            await service.embed_texts(["Annual leave"])
