import os
from typing import cast

from chromadb import QueryResult
import pytest
from chromadbx.reranking import RerankedDocuments, RerankedQueryResult
from chromadbx.reranking.together import TogetherReranker


from unittest.mock import MagicMock

_together = pytest.importorskip("together", reason="together not installed")


def test_together_mock_rerank_documents() -> None:
    mock_client = MagicMock()
    mock_client.rerank.return_value = MagicMock(results=[])

    together = TogetherReranker(api_key="test")
    together._client = mock_client

    queries = "What is the capital of the United States?"
    rerankables = ["Washington, D.C.", "New York", "Los Angeles"]

    together(queries, rerankables)
    mock_client.rerank.create.assert_called_once_with(
        model="Salesforce/Llama-Rank-V1",
        query=queries,
        documents=rerankables,
        top_n=len(rerankables),
    )


@pytest.mark.skipif(
    os.getenv("TOGETHER_API_KEY") is None,
    reason="TOGETHER_API_KEY environment variable is not set",
)
def test_together_rerank_documents() -> None:
    together = TogetherReranker(api_key=os.getenv("TOGETHER_API_KEY", ""))
    queries = "What is the capital of the United States?"
    rerankables = ["Washington, D.C.", "New York", "Los Angeles"]
    result = cast(RerankedDocuments, together(queries, rerankables))
    assert "ranked_distances" in result
    assert len(result["ranked_distances"][together.id()]) == len(rerankables)
    assert result["ranked_distances"][together.id()].index(
        min(result["ranked_distances"][together.id()])
    ) == rerankables.index("Washington, D.C.")


@pytest.mark.skipif(
    os.getenv("TOGETHER_API_KEY") is None,
    reason="TOGETHER_API_KEY environment variable is not set",
)
def test_together_rerank_documents_with_query_result() -> None:
    together = TogetherReranker(api_key=os.getenv("TOGETHER_API_KEY", ""))
    queries = ["What is the capital of the United States?"]
    rerankables = QueryResult(
        documents=[["Washington, D.C.", "New York", "Los Angeles"]],
        metadatas=[[{"source": "test"}, {"source": "test"}, {"source": "test"}]],
        embeddings=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ids=[["id1", "id2", "id3"]],
    )
    result = cast(RerankedQueryResult, together(queries, rerankables))
    assert "ranked_distances" in result
    assert len(result["ranked_distances"][together.id()]) == len(rerankables["ids"][0])
    assert result["ranked_distances"][together.id()].index(
        min(result["ranked_distances"][together.id()])
    ) == rerankables["ids"][0].index("id1")