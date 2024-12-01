import os
from typing import Optional, cast

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction


class MistralAIEmbeddings(EmbeddingFunction[Documents]):  # type: ignore[misc]
    """
    This class is used to get embeddings for a list of texts using the Mistral AI API.
    """

    def __init__(
        self,
        model_name: str = "mistral-embed",
        *,
        api_key: Optional[str] = os.getenv("MISTRAL_API_KEY"),
        retries: Optional[int] = None,
    ):
        """
        Initialize the Mistral AI EF.

        :param model_name: The name of the model to use. Defaults to "mistral-embed".
        :param api_key: The API key
        :param retries: The number of retries to use. Defaults to None.
        """
        try:
            from mistralai import Mistral

            self._client = Mistral(api_key=api_key)
            self._model = model_name
            self._retries = retries
        except ImportError:
            raise ValueError(
                "The mistralai python package is not installed. Please install it with `pip install mistralai`"
            )

    def __call__(self, input: Documents) -> Embeddings:
        embeddings_batch_response = self._client.embeddings.create(
            model=self._model,
            inputs=input,
        )
        embeddings = [d.embedding for d in embeddings_batch_response.data]
        return cast(Embeddings, embeddings)
