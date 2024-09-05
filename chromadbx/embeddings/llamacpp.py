import os.path
from typing import Optional

from chromadb import EmbeddingFunction, Documents


class LlamaCppEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_path: str) -> None:
        try:
            from llama_embedder import LlamaEmbedder, PoolingType
        except ImportError:
            raise ValueError(
                "The cohere python package is not installed. Please install it with `pip install llama-embedder`"
            )

        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")
        self._embedder = LlamaEmbedder(
            model_path=model_path, pooling_type=PoolingType.MEAN
        )

    def __call__(self, input: Documents) -> Optional[Documents]:
        return self._embedder.embed(input)
