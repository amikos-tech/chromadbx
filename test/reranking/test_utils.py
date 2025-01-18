from typing import List
from chromadb import QueryResult
import pytest
from chromadbx.reranking.utils import get_query_documents_tuples


def test_get_query_documents_tuples() -> None:
    queries = "What is the capital of the United States?"
    documents = [
        "What is the capital of the United States?",
        "What is the capital of the United States?",
    ]
    assert get_query_documents_tuples(queries, documents) == [
        (
            "What is the capital of the United States?",
            [
                "What is the capital of the United States?",
                "What is the capital of the United States?",
            ],
        )
    ]


def test_get_query_documents_tuples_list() -> None:
    queries = ["What is the capital of the United States?"]
    documents = QueryResult(
        documents=[
            [
                "What is the capital of the United States?",
                "What is the capital of the United States?",
            ]
        ],
        metadatas=[[{"source": "test"}, {"source": "test"}]],
        ids=[["id1", "id2"]],
    )
    assert get_query_documents_tuples(queries, documents) == [
        (
            "What is the capital of the United States?",
            [
                "What is the capital of the United States?",
                "What is the capital of the United States?",
            ],
        )
    ]


def test_get_query_documents_tuples_dict() -> None:
    with pytest.raises(ValueError):
        get_query_documents_tuples(None, None)  # type: ignore
    with pytest.raises(ValueError):
        get_query_documents_tuples([], None)

    with pytest.raises(ValueError):
        get_query_documents_tuples(None, {})  # type: ignore

    with pytest.raises(ValueError):
        get_query_documents_tuples([], {})

    with pytest.raises(ValueError):
        get_query_documents_tuples([], [])

    with pytest.raises(ValueError):
        queries: List[str] = []
        documents = QueryResult(
            documents=[
                [
                    "What is the capital of the United States?",
                    "What is the capital of the United States?",
                ]
            ],
            metadatas=[[{"source": "test"}, {"source": "test"}]],
            ids=[["id1", "id2"]],
        )
        get_query_documents_tuples(queries, documents)

    with pytest.raises(ValueError):
        queries = ["What is the capital of the United States?"]
        documents = QueryResult(
            documents=[
                ["What is the capital of the United States?"],
                ["What is the capital of the United States?"],
            ],
            metadatas=[[{"source": "test"}], [{"source": "test"}]],
            ids=[["id1"], ["id2"]],
        )
        get_query_documents_tuples(queries, documents)

    with pytest.raises(ValueError):
        queries = [
            "What is the capital of the United States?",
            "What is the capital of the United States?",
        ]
        documents = QueryResult(
            documents=[["What is the capital of the United States?"]],
            metadatas=[[{"source": "test"}]],
            ids=[["id1"]],
        )
        get_query_documents_tuples(queries, documents)
