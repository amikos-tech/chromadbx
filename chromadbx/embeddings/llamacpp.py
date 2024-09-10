import os.path
from enum import Enum
from typing import Optional

from chromadb import EmbeddingFunction, Documents


class PoolingType(int, Enum):
    NONE = 0
    MEAN = 1
    CLS = 2
    LAST = 3


class LlamaCppEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model_path: str,
        *,
        hf_file_name: Optional[str] = None,
        pooling_type: Optional[PoolingType] = PoolingType.MEAN,
    ) -> None:
        """
        Initialize the LlamaCppEmbeddingFunction.

        :param model_path: This can be a local path to the model or the HuggingFace repository. You need to install huggingface_hub package.
        :param hf_file_name: The name of the file in the HuggingFace repository.
            This is only required if the model_path is a HuggingFace repository.
        :param pooling_type: The pooling type to use. Default is `PoolingType.MEAN`.
        """
        try:
            from llama_embedder import LlamaEmbedder, PoolingType as PT
        except ImportError:
            raise ValueError(
                "The `llama-embedder` python package is not installed. "
                "Please install it with `pip install llama-embedder`"
            )

        if not os.path.exists(model_path) and hf_file_name is None:
            raise ValueError(f"Model path {model_path} does not exist")
        elif os.path.exists(model_path):
            self._model_file = model_path
        elif model_path and hf_file_name:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise ValueError(
                    "The `huggingface_hub` python package is not installed. "
                    "Please install it with `pip install huggingface_hub`"
                )
            self._model_file = hf_hub_download(
                repo_id=model_path, filename=hf_file_name
            )
        if pooling_type is None:
            pt = PT.NONE
        elif pooling_type == PoolingType.MEAN:
            pt = PT.MEAN
        elif pooling_type == PoolingType.CLS:
            pt = PT.CLS
        elif pooling_type == PoolingType.LAST:
            pt = PT.LAST
        else:
            raise ValueError(f"Invalid pooling type: {pooling_type}")

        self._embedder = LlamaEmbedder(model_path=self._model_file, pooling_type=pt)

    def __call__(self, input: Documents) -> Optional[Documents]:
        return self._embedder.embed(input)
