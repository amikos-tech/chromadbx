import os
import pytest
from chromadbx.embeddings.mistral import MistralAiEmbeddings

vai = pytest.importorskip("mistralai", reason="mistralai not installed")


@pytest.mark.skipif(
    "MISTRAL_API_KEY" not in os.environ,
    reason="MISTRAL_API_KEY not set, skipping test.",
)
def test_embed() -> None:
    ef = MistralAiEmbeddings()
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
    assert len(embeddings[1]) == 1024
