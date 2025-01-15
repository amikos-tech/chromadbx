import pytest
from chromadbx.embeddings.spacy import SpacyEmbeddingFunction
import subprocess

spacy = pytest.importorskip("spacy", reason="spacy not installed")


def download_model(model_name: str) -> None:
    s = subprocess.run(["python", "-m", "spacy", "download", model_name])
    assert s.returncode == 0


def test_spacy() -> None:
    download_model("en_core_web_lg")
    ef = SpacyEmbeddingFunction()
    texts = ["hello world", "goodbye world"] * 1000
    embeddings = ef(texts)
    assert embeddings is not None
    assert len(embeddings) == 2000
    assert len(embeddings[0]) == 300
    assert len(embeddings[1]) == 300


def test_spacy_with_model() -> None:
    download_model("en_core_web_sm")
    ef = SpacyEmbeddingFunction(model_name="en_core_web_sm")
    embeddings = ef(["hello world", "goodbye world"])
    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 96
    assert len(embeddings[1]) == 96


def test_spacy_with_invalid_model() -> None:
    with pytest.raises(ValueError) as e:
        SpacyEmbeddingFunction(model_name="invalid_model")
    assert "spacy model 'invalid_model' are not downloaded yet" in str(e.value)
