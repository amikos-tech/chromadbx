import os
import pytest
from chromadbx.embeddings.together import TogetherEmbeddingFunction

together = pytest.importorskip("together", reason="together not installed")

@pytest.mark.skipif(
    os.getenv("TOGETHER_API_KEY") is None,
    reason="TOGETHER_API_KEY environment variable is not set"
)
def test_together() -> None:
    ef = TogetherEmbeddingFunction(api_key=os.getenv("TOGETHER_API_KEY"))
    texts = ["hello world", "goodbye world"]
    embeddings = ef(texts)
    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536