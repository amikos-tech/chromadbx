import logging
from typing import Optional, cast

import numpy as np
import numpy.typing as npt
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class GoogleVertexAiEmbeddings(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model_name: str = "text-embedding-004",
        *,
        project_id: Optional[str] = None,
        location: Optional[str] = "us-central1",
        dimensions: Optional[int] = 256,
        task_type: Optional[str] = "RETRIEVAL_DOCUMENT",
    ) -> None:
        """
        Initialize the OnnxRuntimeEmbeddings.

        :param model_name: The name of the model to use. Defaults to "text-embedding-004".
        :param project_id: The project ID to use. Defaults to None.
        :param location: The location to use. Defaults to None.
        :param dimensions: The number of dimensions to use. Defaults to None.
        :param task_type: The task type to use. Defaults to "RETRIEVAL_DOCUMENT". https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types

        """
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
            if project_id and location:
                vertexai.init(project=project_id, location=location)
            elif project_id and not location:
                vertexai.init(project=project_id)
            elif not project_id and location:
                vertexai.init(location=location)
            else:
                vertexai.init()
            self._model = TextEmbeddingModel.from_pretrained(model_name)
        except ImportError:
            raise ValueError(
                "The vertexai python package is not installed. Please install it with `pip install vertexai`"
            )
        self._dimensions = dimensions
        self._task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        from vertexai.language_models import TextEmbeddingInput
        inputs = [TextEmbeddingInput(text, self._task_type) for text in input]
        kwargs = dict(output_dimensionality=self._dimensions) if self._dimensions else {}
        embeddings = [embedding.values for embedding in self._model.get_embeddings(inputs,**kwargs)]
        return cast(Embeddings, embeddings)
