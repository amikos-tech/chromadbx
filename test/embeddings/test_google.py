from chromadbx.embeddings.google import GoogleVertexAiEmbeddings


def test_embed() -> None:
    ef = GoogleVertexAiEmbeddings()
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 256
    assert len(embeddings[1]) == 256
    assert embeddings[0] != embeddings[1]

def test_with_model() -> None:
    ef = GoogleVertexAiEmbeddings(
        model_name="text-multilingual-embedding-002",
    )
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 256
    assert len(embeddings[1]) == 256
    assert embeddings[0] != embeddings[1]

def test_dimensions() -> None:
    ef = GoogleVertexAiEmbeddings(
        model_name="text-multilingual-embedding-002",
        dimensions=768,
    )
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    assert len(embeddings[1]) == 768
    assert embeddings[0] != embeddings[1]

def test_task_type() -> None:
    ef = GoogleVertexAiEmbeddings(
        task_type="RETRIEVAL_QUERY",
    )
    embeddings = ef(["hello world", "goodbye world"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 256
    assert len(embeddings[1]) == 256