import importlib
import logging
import os
from functools import cached_property
from typing import List, Optional, cast

import numpy as np
import numpy.typing as npt
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class OnnxRuntimeEmbeddings(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model_path,
        *,
        preferred_providers: Optional[List[str]] = None,
        max_length: Optional[int] = 256,
        enabled_padding: Optional[bool] = True,
    ) -> None:
        # Import dependencies on demand to mirror other embedding functions. This
        # breaks typechecking, thus the ignores.
        # convert the list to set for unique values
        if preferred_providers and not all(
            [isinstance(i, str) for i in preferred_providers]
        ):
            raise ValueError("Preferred providers must be a list of strings")
        # check for duplicate providers
        if preferred_providers and len(preferred_providers) != len(
            set(preferred_providers)
        ):
            raise ValueError("Preferred providers must be unique")
        self._preferred_providers = preferred_providers
        self._model_path = os.path.expanduser(model_path)
        print(self._model_path)
        self._max_length = max_length
        self._enabled_padding = enabled_padding
        try:
            # Equivalent to import onnxruntime
            self.ort = importlib.import_module("onnxruntime")
        except ImportError:
            raise ValueError(
                "The onnxruntime python package is not installed. Please install it with `pip install onnxruntime`"
            )
        try:
            self.Tokenizer = importlib.import_module("tokenizers").Tokenizer
        except ImportError:
            raise ValueError(
                "The tokenizers python package is not installed. Please install it with `pip install tokenizers`"
            )

    # Use pytorches default epsilon for division by zero
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    def _normalize(self, v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        norm = np.linalg.norm(v, axis=1)
        norm[norm == 0] = 1e-12
        return cast(npt.NDArray[np.float32], v / norm[:, np.newaxis])

    def _forward(
        self, documents: List[str], batch_size: int = 32
    ) -> npt.NDArray[np.float32]:
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            input_names = [input.name for input in self.model.get_inputs()]
            batch = documents[i : i + batch_size]
            encoded = [self.tokenizer.encode(d) for d in batch]
            input_ids = np.array([e.ids for e in encoded])
            attention_mask = np.array([e.attention_mask for e in encoded])
            onnx_input = {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "token_type_ids": np.array(
                    [np.zeros(len(e), dtype=np.int64) for e in input_ids],
                    dtype=np.int64,
                ),
            }
            if "token_type_ids" not in input_names:
                del onnx_input["token_type_ids"]
            model_output = self.model.run(None, onnx_input)
            last_hidden_state = model_output[0]
            # Perform mean pooling with attention weighting
            input_mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask, -1), last_hidden_state.shape
            )
            embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(
                input_mask_expanded.sum(1), a_min=1e-9, a_max=None
            )
            embeddings = self._normalize(embeddings).astype(np.float32)
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings)

    @cached_property
    def tokenizer(self) -> "Tokenizer":  # noqa F821
        tokenizer = self.Tokenizer.from_file(
            os.path.join(self._model_path, "tokenizer.json")
        )
        if self._max_length:
            tokenizer.enable_truncation(max_length=self._max_length)
        if self._enabled_padding:
            tokenizer.enable_padding(length=self._max_length)
        # tokenizer.enable_truncation(max_length=256)
        # tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
        return tokenizer

    @cached_property
    def model(self) -> "InferenceSession":  # noqa F821
        print(self.ort.get_available_providers())
        if self._preferred_providers is None or len(self._preferred_providers) == 0:
            if len(self.ort.get_available_providers()) > 0:
                logger.debug(
                    f"WARNING: No ONNX providers provided, defaulting to available providers: "
                    f"{self.ort.get_available_providers()}"
                )
            self._preferred_providers = self.ort.get_available_providers()
        elif not set(self._preferred_providers).issubset(
            set(self.ort.get_available_providers())
        ):
            raise ValueError(
                f"Preferred providers must be subset of available providers: {self.ort.get_available_providers()}"
            )

        so = self.ort.SessionOptions()
        _model_paths = [
            os.path.join(self._model_path, "model.onnx"),
            os.path.join(self._model_path, "onnx", "model.onnx"),
        ]
        _actual_model_path = None
        # possible paths for model model_path/model.onnx or model_path/onnx/model.onnx
        for mp in _model_paths:
            if os.path.exists(mp):
                _actual_model_path = mp
                break
        if not _actual_model_path:
            raise ValueError(
                f"Cannot find onnx model un the following paths: {_model_paths}"
            )

        sess = self.ort.InferenceSession(
            _actual_model_path,
            # Since 1.9 onnyx runtime requires providers to be specified when there are multiple available - https://onnxruntime.ai/docs/api/python/api_summary.html
            # This is probably not ideal but will improve DX as no exceptions will be raised in multi-provider envs
            providers=self._preferred_providers,
            sess_options=so,
        )

        return sess

    def __call__(self, input: Documents) -> Embeddings:
        return cast(Embeddings, self._forward(input).tolist())
