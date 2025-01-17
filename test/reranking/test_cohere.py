
import os

import pytest
from chromadbx.reranking.cohere import CohereReranker


from unittest.mock import MagicMock

_cohere = pytest.importorskip("cohere", reason="cohere not installed")

def test_cohere_mock_rerank_documents()->None:
    mock_client = MagicMock()
    mock_client.rerank.return_value = MagicMock(results=[])
    
    cohere = CohereReranker(api_key="test")
    cohere._client = mock_client
    
    queries = "What is the capital of the United States?"
    rerankables = ["Washington, D.C.", "New York", "Los Angeles"]
    
    cohere(queries, rerankables)
    mock_client.rerank.assert_called_once_with(
        model="rerank-v3.5",
        query=queries,
        documents=rerankables,
        top_n=len(rerankables),
        max_tokens_per_doc=4096,
        request_options=cohere._request_options,
    )

@pytest.mark.skipif(
    os.getenv("COHERE_API_KEY") is None,
    reason="COHERE_API_KEY environment variable is not set",
)
def test_cohere_rerank_documents()->None:
    cohere = CohereReranker(api_key=os.getenv("COHERE_API_KEY"))
    queries = "What is the capital of the United States?"
    rerankables = ["Washington, D.C.", "New York", "Los Angeles"]
    result = cohere(queries, rerankables)
    print(result)
