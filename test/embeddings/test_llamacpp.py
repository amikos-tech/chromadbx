import os
from chromadbx.embeddings.llamacpp import LlamaCppEmbeddingFunction
import pytest
from huggingface_hub import hf_hub_download

DEFAULT_REPO = "leliuga/all-MiniLM-L6-v2-GGUF"
DEFAULT_TEST_MODEL = "all-MiniLM-L6-v2.Q4_0.gguf"


@pytest.fixture(scope="module")
def get_model() -> str:
    hf_hub_download(
        repo_id=DEFAULT_REPO, filename=DEFAULT_TEST_MODEL, local_dir="./local_model"
    )
    return os.path.join("local_model", DEFAULT_TEST_MODEL)


def test_embed(get_model: str) -> None:
    ef = LlamaCppEmbeddingFunction(model_path=get_model)
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


def test_embed_from_hf_model() -> None:
    ef = LlamaCppEmbeddingFunction(
        model_path=DEFAULT_REPO, hf_file_name=DEFAULT_TEST_MODEL
    )
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
