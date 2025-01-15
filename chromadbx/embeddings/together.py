from typing import Optional, cast

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction


class TogetherEmbeddingFunction(EmbeddingFunction[Documents]):  # type: ignore[misc]
    """
    This class is used to get embeddings for a list of texts from together's embedding models.
    It requires an API key and a model name. The default model name is "togethercomputer/m2-bert-80M-8k-retrieval".
    For more, refer to the official documentation at "https://docs.together.ai/docs/embeddings-python".
    """

    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = "togethercomputer/m2-bert-80M-8k-retrieval",
    ):
        """
        Initialize the TogetherEmbeddingFunction.

        Args:
            api_key (str): The API key for the Together API.
            model_name (Optional[str]): The name of the model to use for embedding. Defaults to "togethercomputer/m2-bert-80M-8k-retrieval".
        """

        try:
            import together
        except ImportError:
            raise ValueError(
                "The together python package is not installed. Please install it with `pip install together`"
            )
        together.api_key = api_key
        self.model_name = model_name
        self.client = together.Together()

    def __call__(self, input: Documents) -> Embeddings:
        """
        Get the embeddings for a list of texts.

        Args:
            input (Documents): A list of texts to get embeddings for.

        Returns:
            Embeddings: The embeddings for the texts.

        Example:
        ```python
        import os
        from chromadbx.embeddings.together import TogetherEmbeddingFunction

        ef = TogetherEmbeddingFunction(api_key=os.getenv("TOGETHER_API_KEY"))
        embeddings = ef(["hello world", "goodbye world"])
        ```
        """
        outputs = self.client.embeddings.create(input=input, model=self.model_name)
        return cast(Embeddings, [outputs.data[i].embedding for i in range(len(input))])
