import os
import pytest
from chromadbx.embeddings.google import GoogleVertexAiEmbeddings

vai = pytest.importorskip("vertexai", reason="vertexai not installed")


def test_embed() -> None:
    ef = GoogleVertexAiEmbeddings()
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 256
    assert len(embeddings[1]) == 256


def test_with_model() -> None:
    ef = GoogleVertexAiEmbeddings(
        model_name="text-multilingual-embedding-002",
    )
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 256
    assert len(embeddings[1]) == 256


def test_dimensions() -> None:
    ef = GoogleVertexAiEmbeddings(
        model_name="text-multilingual-embedding-002",
        dimensions=768,
    )
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    assert len(embeddings[1]) == 768


def test_task_type() -> None:
    ef = GoogleVertexAiEmbeddings(
        task_type="RETRIEVAL_QUERY",
    )
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 256
    assert len(embeddings[1]) == 256


def test_credentials() -> None:
    file_path = "genai-sa-key.json"
    if not os.path.exists(file_path):
        pytest.skip(f"File {file_path} does not exist")
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(file_path)
    ef = GoogleVertexAiEmbeddings(credentials=credentials)
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 256
    assert len(embeddings[1]) == 256
