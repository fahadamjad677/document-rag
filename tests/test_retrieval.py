
import math

import pytest

from app.schemas import PageText
from app.services.retrieval import (
    cosine_similarity,
    retrieve_top_k,
)
from app.services.text_chunker import chunk_pages


def make_chunk(text: str, page: int = 1):
    chunks = chunk_pages(
        [PageText(page_number=page, text=text)],
        chunk_size=1000,
        overlap=0,
    )
    assert len(chunks) == 1
    return chunks[0]


def test_identical_vectors_have_similarity_one() -> None:
    vector = [1.0, 2.0, 3.0]

    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    assert cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    ) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one() -> None:
    assert cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    ) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("vector_a", "vector_b"),
    [
        ([], [1.0]),
        ([1.0], []),
        ([1.0], [1.0, 2.0]),
        ([0.0, 0.0], [1.0, 2.0]),
        ([1.0], [0.0]),
        ([math.inf], [1.0]),
        ([1.0], [math.nan]),
    ],
)
def test_invalid_vectors_are_rejected(
    vector_a: list[float],
    vector_b: list[float],
) -> None:
    with pytest.raises(ValueError):
        cosine_similarity(vector_a, vector_b)


def test_retrieve_top_k_returns_best_matches_first() -> None:
    chunks = [
        make_chunk("Unrelated inventory passage."),
        make_chunk("Annual leave policy."),
        make_chunk("Annual leave entitlement details."),
    ]

    query_embedding = [1.0, 0.0]

    embeddings = [
        [0.0, 1.0],
        [0.8, 0.2],
        [1.0, 0.0],
    ]

    results = retrieve_top_k(
        query_embedding,
        chunks,
        embeddings,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.text == "Annual leave entitlement details."
    assert results[0].similarity == pytest.approx(1.0)
    assert results[1].chunk.text == "Annual leave policy."


def test_top_k_cannot_be_zero_or_negative() -> None:
    with pytest.raises(ValueError, match="top_k"):
        retrieve_top_k(
            [1.0, 0.0],
            [make_chunk("Example.")],
            [[1.0, 0.0]],
            top_k=0,
        )


def test_chunk_and_embedding_counts_must_match() -> None:
    with pytest.raises(ValueError, match="exactly one embedding"):
        retrieve_top_k(
            [1.0, 0.0],
            [make_chunk("Example.")],
            [],
        )


def test_retrieval_preserves_source_page_metadata() -> None:
    chunks = [make_chunk("Leave policy.", page=7)]

    results = retrieve_top_k(
        [1.0, 0.0],
        chunks,
        [[1.0, 0.0]],
    )

    assert results[0].chunk.page_number == 7
    assert results[0].chunk.text == "Leave policy."
