
import math
from typing import TypeAlias
from urllib.parse import urlsplit

import httpx


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

EmbeddingVector: TypeAlias = list[float]
EmbeddingVectors: TypeAlias = list[EmbeddingVector]


class EmbeddingServiceError(Exception):
    """Base exception for embedding service failures."""


class EmbeddingServiceTimeoutError(EmbeddingServiceError):
    """Raised when the embedding request times out."""


class EmbeddingServiceConnectionError(EmbeddingServiceError):
    """Raised when the embedding service cannot be reached."""


class EmbeddingServiceHTTPError(EmbeddingServiceError):
    """Raised when Ollama returns an unsuccessful HTTP status."""


class EmbeddingServiceResponseError(EmbeddingServiceError):
    """Raised when Ollama returns an invalid response."""


class EmbeddingService:
    """Generate and validate text embeddings using Ollama."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_EMBEDDING_MODEL,
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(base_url, str):
            raise ValueError("Ollama base URL must be a string.")

        if not isinstance(model, str):
            raise ValueError("Embedding model must be a string.")

        normalized_base_url = base_url.strip().rstrip("/")
        normalized_model = model.strip()

        if not normalized_base_url:
            raise ValueError("Ollama base URL cannot be empty.")

        parsed_url = urlsplit(normalized_base_url)

        try:
            valid_port = parsed_url.port is None or (
                1 <= parsed_url.port <= 65535
            )
        except ValueError:
            valid_port = False

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
            or not valid_port
        ):
            raise ValueError(
                "Ollama base URL must be a valid HTTP(S) URL."
            )

        if not normalized_model:
            raise ValueError("Embedding model cannot be empty.")

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "Timeout must be a finite positive number."
            )

        if client is not None and not isinstance(
            client, httpx.AsyncClient
        ):
            raise ValueError(
                "Client must be an httpx.AsyncClient instance."
            )

        self._base_url = normalized_base_url
        self._model = normalized_model
        self._timeout_seconds = float(timeout_seconds)
        self._client = client

    async def embed_texts(
        self,
        texts: list[str],
    ) -> EmbeddingVectors:
        """Return one validated embedding vector per input text."""

        self._validate_inputs(texts)

        if not texts:
            return []

        payload = {
            "model": self._model,
            "input": texts,
        }

        try:
            if self._client is not None:
                response = await self._client.post(
                    f"{self._base_url}/api/embed",
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self._base_url}/api/embed",
                        json=payload,
                        timeout=self._timeout_seconds,
                    )

        except httpx.TimeoutException as exc:
            raise EmbeddingServiceTimeoutError(
                "Ollama embedding request timed out."
            ) from exc

        except httpx.RequestError as exc:
            raise EmbeddingServiceConnectionError(
                "Could not connect to Ollama. "
                "Check that Ollama is running."
            ) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingServiceHTTPError(
                f"Ollama returned HTTP {exc.response.status_code}. "
                "Check that the model is installed and Ollama is running."
            ) from exc

        data = self._parse_json(response)

        return self._validate_response(
            data=data,
            expected_count=len(texts),
        )

    @staticmethod
    def _validate_inputs(texts: list[str]) -> None:
        if not isinstance(texts, list):
            raise ValueError(
                "Input texts must be a list of strings."
            )

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Input at index {index} must be a string."
                )

            if not text.strip():
                raise ValueError(
                    f"Input text at index {index} cannot be empty."
                )

    @staticmethod
    def _parse_json(response: httpx.Response) -> object:
        try:
            # Allow non-standard JSON constants to be parsed as floats.
            # Vector validation below will reject NaN and infinities.
            return response.json(
                parse_constant=lambda value: float(value)
            )
        except (ValueError, UnicodeDecodeError) as exc:
            raise EmbeddingServiceResponseError(
                "Ollama returned an invalid JSON response."
            ) from exc

    @classmethod
    def _validate_response(
        cls,
        *,
        data: object,
        expected_count: int,
    ) -> EmbeddingVectors:
        if not isinstance(data, dict):
            raise EmbeddingServiceResponseError(
                "Ollama returned an invalid response object."
            )

        raw_embeddings = data.get("embeddings")

        if not isinstance(raw_embeddings, list):
            raise EmbeddingServiceResponseError(
                "Ollama returned an invalid embeddings response."
            )

        if len(raw_embeddings) != expected_count:
            raise EmbeddingServiceResponseError(
                "The number of returned embeddings does not "
                "match the number of input texts."
            )

        embeddings: EmbeddingVectors = []

        for vector_index, raw_vector in enumerate(raw_embeddings):
            vector = cls._validate_vector(
                raw_vector,
                vector_index=vector_index,
            )
            embeddings.append(vector)

        # Empty input is handled before the HTTP request.
        # For non-empty input, the count check guarantees a vector exists.
        dimensions = len(embeddings[0])

        if any(
            len(vector) != dimensions
            for vector in embeddings
        ):
            raise EmbeddingServiceResponseError(
                "Embedding vectors have inconsistent dimensions."
            )

        return embeddings

    @staticmethod
    def _validate_vector(
        raw_vector: object,
        *,
        vector_index: int,
    ) -> EmbeddingVector:
        if not isinstance(raw_vector, list) or not raw_vector:
            raise EmbeddingServiceResponseError(
                f"Embedding vector {vector_index} is empty or invalid."
            )

        vector: EmbeddingVector = []

        for value in raw_vector:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise EmbeddingServiceResponseError(
                    "Embedding values must be numeric."
                )

            try:
                numeric_value = float(value)
            except OverflowError as exc:
                raise EmbeddingServiceResponseError(
                    "Embedding values must be finite."
                ) from exc

            if not math.isfinite(numeric_value):
                raise EmbeddingServiceResponseError(
                    "Embedding values must be finite."
                )

            vector.append(numeric_value)

        return vector
