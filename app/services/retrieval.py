
import math

from pydantic import BaseModel, ConfigDict, Field

from app.services.text_chunker import TextChunk


class RetrievedChunk(BaseModel):
    """A chunk ranked by its similarity to a query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: TextChunk
    similarity: float = Field(ge=-1.0, le=1.0)


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two nonzero vectors."""

    if not vector_a or not vector_b:
        raise ValueError("Vectors cannot be empty.")

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions.")

    if any(not math.isfinite(value) for value in vector_a):
        raise ValueError("Vector A must contain finite values.")

    if any(not math.isfinite(value) for value in vector_b):
        raise ValueError("Vector B must contain finite values.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b, strict=True)
    )

    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return dot_product / (norm_a * norm_b)


def retrieve_top_k(
    query_embedding: list[float],
    chunks: list[TextChunk],
    chunk_embeddings: list[list[float]],
    *,
    top_k: int = 3,
) -> list[RetrievedChunk]:
    """
    Rank chunks by cosine similarity to a query embedding.

    Embeddings must correspond to chunks in the same order.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if len(chunks) != len(chunk_embeddings):
        raise ValueError(
            "Each chunk must have exactly one embedding."
        )

    if not chunks:
        return []

    # Validate the query even if no chunks are available.
    if not query_embedding:
        raise ValueError("Query embedding cannot be empty.")

    if any(not math.isfinite(v) for v in query_embedding):
        raise ValueError("Query embedding must contain finite values.")

    if all(value == 0.0 for value in query_embedding):
        raise ValueError("Query embedding cannot be a zero vector.")

    scored_chunks: list[RetrievedChunk] = []

    for chunk, embedding in zip(
        chunks,
        chunk_embeddings,
        strict=True,
    ):
        score = cosine_similarity(query_embedding, embedding)

        scored_chunks.append(
            RetrievedChunk(
                chunk=chunk,
                similarity=max(-1.0, min(1.0, score)),
            )
        )

    # Stable sorting preserves original chunk order when scores tie.
    scored_chunks.sort(
        key=lambda result: result.similarity,
        reverse=True,
    )

    return scored_chunks[:top_k]
