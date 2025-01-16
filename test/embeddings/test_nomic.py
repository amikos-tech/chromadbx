import os
import pytest
from chromadbx.embeddings.nomic import NomicEmbeddingFunction

httpx = pytest.importorskip("httpx", reason="nomic not installed")

@pytest.mark.skipif(
    os.getenv("NOMIC_API_KEY") is None,
    reason="NOMIC_API_KEY environment variable is not set",
)
def test_nomic() -> None:
    ef = NomicEmbeddingFunction()
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    assert len(embeddings[1]) == 768

@pytest.mark.skipif(
    os.getenv("NOMIC_API_KEY") is None,
    reason="NOMIC_API_KEY environment variable is not set",
)
def test_nomic_with_api_key() -> None:
    ef = NomicEmbeddingFunction(api_key=os.getenv("NOMIC_API_KEY"))
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    assert len(embeddings[1]) == 768

@pytest.mark.skipif(
    os.getenv("NOMIC_API_KEY") is None,
    reason="NOMIC_API_KEY environment variable is not set",
)
def test_dimensionality() -> None:
    ef = NomicEmbeddingFunction(dimensionality=512)
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 512
    assert len(embeddings[1]) == 512
