import os
from chromadbx.embeddings.onnx import OnnxRuntimeEmbeddings
import pytest
from huggingface_hub import hf_hub_download

DEFAULT_REPO = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_FILES = [
    "config.json",
    "onnx/model.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]


@pytest.fixture(scope="module")
def get_model() -> str:
    for file in DEFAULT_MODEL_FILES:
        if not os.path.exists(os.path.join("local_model", "all-MiniLM-L6-v2", file)):
            hf_hub_download(
                repo_id=DEFAULT_REPO,
                filename=file,
                local_dir="./local_model/all-MiniLM-L6-v2",
            )
    return os.path.join("local_model", "all-MiniLM-L6-v2")


def test_download(get_model):
    print(get_model)
    ef = OnnxRuntimeEmbeddings(
        model_path=get_model, preferred_providers=["CPUExecutionProvider"]
    )
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
