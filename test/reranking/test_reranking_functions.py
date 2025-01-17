import pytest
from chromadbx.reranking import validate_rerankables
from chromadb import QueryResult, Documents

def test_validate_rerankables_valid_query_result():
    queries = ["What is the capital of the United States?"]
    rerankables = QueryResult(
        documents=[["Washington, D.C.", "New York", "Los Angeles"]],
        metadatas=[[{"source": "test"}, {"source": "test"}, {"source": "test"}]],
        ids=[["id1", "id2", "id3"]],
    )
    validate_rerankables(queries, rerankables)

def test_validate_rerankables_valid_documents():
    queries = "What is the capital of the United States?"
    rerankables = ["Washington, D.C.", "New York", "Los Angeles"]
    validate_rerankables(queries, rerankables)

def test_validate_rerankables_invalid_queries():
    with pytest.raises(ValueError):
        validate_rerankables(None, ["Washington, D.C."])

def test_validate_rerankables_empty_rerankables():
    with pytest.raises(ValueError):
        validate_rerankables("What is the capital of the United States?", [])

def test_validate_rerankables_invalid_documents():
    with pytest.raises(ValueError):
        validate_rerankables("What is the capital of the United States?", [123, "New York"])

def test_validate_rerankables_mismatched_queries_and_documents():
    queries = ["What is the capital of the United States?"]
    rerankables = QueryResult(
        documents=[["Washington, D.C."], ["New York"]],
        metadatas=[[{"source": "test"}], [{"source": "test"}]],
        ids=[["id1"], ["id2"]],
    )
    with pytest.raises(ValueError):
        validate_rerankables(queries, rerankables)
