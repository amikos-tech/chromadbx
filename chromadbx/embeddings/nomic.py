from enum import Enum
import os
from typing import Optional, cast

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction


class TaskType(str, Enum):
    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class LongTextMode(str, Enum):
    TRUNCATE = "truncate"
    MEAN = "mean"


class NomicEmbeddingFunction(EmbeddingFunction[Documents]):  # type: ignore[misc]
    """
    Nomic Embedding Function using the Nomic Embedding API - https://docs.nomic.ai/atlas/models/text-embedding.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = "nomic-embed-text-v1.5",
        *,
        dimensionality: Optional[int] = 768,
        max_tokens_per_text: Optional[int] = 8192,
        long_text_mode: Optional[LongTextMode] = LongTextMode.TRUNCATE,
        task_type: Optional[TaskType] = TaskType.SEARCH_DOCUMENT,
        timeout: Optional[float] = 60.0,
    ) -> None:
        """
        Initialize the Nomic Embedding Function.

        Read more about the Nomic Embedding API here: https://docs.nomic.ai/reference/api/embed-text-v-1-embedding-text-post#request

        Args:
            api_key (str): The API key to use for the Nomic Embedding API.
            model_name (str): The name of the model to use for text embeddings. E.g. "nomic-embed-text-v1.5" (see https://docs.nomic.ai/atlas/models/text-embedding for available models).
            dimensionality (int): The dimensionality of the embeddings. E.g. 768 for "nomic-embed-text-v1.5".
            max_tokens_per_text (int): The maximum number of tokens per text. E.g. 8192 for "nomic-embed-text-v1.5".
            long_text_mode (str): The mode to use for long texts. E.g. "truncate" or "mean".
            task_type (str): The task type to use for the Nomic Embedding API. E.g. "search_document", "search_query", "classification", and "clustering".
            timeout (float): The timeout for the Nomic Embedding API. E.g. 60.0 for 60 seconds.
        """
        try:
            import httpx
        except ImportError:
            raise ValueError(
                "The httpx python package is not installed. Please install it with `pip install httpx`"
            )

        if not api_key and os.getenv("NOMIC_API_KEY") is None:
            raise ValueError(
                "No Nomic API key provided or NOMIC_API_KEY environment variable is not set"
            )
        if not api_key:
            api_key = os.getenv("NOMIC_API_KEY")

        self._api_url = "https://api-atlas.nomic.ai/v1/embedding/text"
        self._model_name = model_name
        self._task_type = task_type
        self._dimensionality = dimensionality
        self._long_text_mode = long_text_mode
        self._max_tokens_per_text = max_tokens_per_text
        self._client = httpx.Client(timeout=timeout)
        self._client.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )

    def __call__(self, input: Documents) -> Embeddings:
        """
        Get the embeddings for a list of texts.

        Args:
            input (Documents): A list of texts to get embeddings for.

        Returns:
            Embeddings: The embeddings for the texts.

        Example:
            >>> from chromadbx.embeddings.nomic import NomicEmbeddingFunction
            >>> nomic_ef = NomicEmbeddingFunction(model_name="nomic-embed-text-v1.5")
            >>> texts = ["Hello, world!", "How are you?"]
            >>> embeddings = nomic_ef(texts)
        """
        texts = input if isinstance(input, list) else [input]

        response = self._client.post(
            self._api_url,
            json={
                "model": self._model_name,
                "texts": texts,
                "task_type": self._task_type.value if self._task_type else None,
                "dimensionality": self._dimensionality,
                "long_text_mode": self._long_text_mode.value
                if self._long_text_mode
                else None,
                "max_tokens_per_text": self._max_tokens_per_text,
            },
        )
        response.raise_for_status()
        response_json = response.json()
        if "embeddings" not in response_json:
            raise RuntimeError("Nomic API did not return embeddings")

        return cast(Embeddings, response_json["embeddings"])
