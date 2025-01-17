
import os
from typing import Any, Dict, Optional

from chromadb import Documents, QueryResult
from chromadbx.reranking import Queries, RankedResults, Rerankable, RerankedDocuments, RerankedQueryResult, RerankerID, RerankingFunction
from chromadbx.reranking.utils import get_query_documents_tuples


class CohereReranker(RerankingFunction[Rerankable, RankedResults]):
    def __init__(self, 
                 api_key: str,
                 model_name: Optional[str] = "rerank-v3.5",
                 *,
                 raw_scores: bool = False,
                 top_n: Optional[int] = None,
                 max_tokens_per_document: Optional[int] = 4096,
                 timeout: Optional[int] = 60,
                 max_retries: Optional[int] = 3,
                 additional_headers: Optional[Dict[str, Any]] = None,
                 ):
        try:
            import cohere
            from cohere.core.request_options import RequestOptions
        except ImportError:
            raise ImportError("cohere is not installed. Please install it with `pip install cohere`")
        if not api_key and not os.getenv("COHERE_API_KEY"):
            raise ValueError("API key is required. Please set the COHERE_API_KEY environment variable or pass it directly.")
        if not model_name:
            raise ValueError("Model name is required. Please set the model_name parameter or use the default value.")
        self._client = cohere.ClientV2(api_key)
        self._model_name = model_name
        self._top_n = top_n
        self._raw_scores = raw_scores
        self._max_tokens_per_document = max_tokens_per_document
        self._request_options = RequestOptions(
            timeout_in_seconds=timeout,
            max_retries=max_retries,
            additional_headers=additional_headers,
        )
    def id(self) -> RerankerID:
        return RerankerID("cohere")
    
    def _combine_reranked_results(self,results: "cohere.v2.types.V2RerankResponse", rerankables: Rerankable) -> RankedResults:
        """
        {
  "results": [
    {
      "index": 3,
      "relevance_score": 0.999071
    },
    {
      "index": 4,
      "relevance_score": 0.7867867
    },
    {
      "index": 0,
      "relevance_score": 0.32713068
    }
  ],
  "id": "07734bd2-2473-4f07-94e1-0d9f0e6843cf",
  "meta": {
    "api_version": {
      "version": "2",
      "is_experimental": false
    },
    "billed_units": {
      "search_units": 1
    }
  }
}
        """

        
        if self._raw_scores:
            ordered_scores = [r.relevance_score for r in sorted(results.results, key=lambda x: x.index)]
        else: # by default we calculate the distance to make results comparable with Chroma distance
            ordered_scores = [1-r.relevance_score for r in sorted(results.results, key=lambda x: x.index)]
        if isinstance(rerankables, dict):
            if len(rerankables.ids) != len(ordered_scores):
                ordered_scores = ordered_scores + [None] * (len(rerankables.ids) - len(ordered_scores))
            return RerankedQueryResult(
                ids=rerankables.ids,
                embeddings=rerankables.embeddings,
                documents=rerankables.documents,
                uris=rerankables.uris,
                data=rerankables.data,
                metadatas=rerankables.metadatas,
                distances=rerankables.distances,
                included=rerankables.included,
                ranked_distances={self.id(): ordered_scores},
            )
        elif isinstance(rerankables, list):
            if len(rerankables) != len(ordered_scores):
                ordered_scores = ordered_scores + [None] * (len(rerankables) - len(ordered_scores))
            return RerankedDocuments(
                documents=rerankables,
                ranked_distances={self.id(): ordered_scores},
            )
    
    def __call__(self, queries: Queries, rerankables: Rerankable) -> RankedResults:
        query_documents_tuples = get_query_documents_tuples(queries, rerankables)
        for query, documents in query_documents_tuples:
            response = self._client.rerank(
                model=self._model_name,
                query=query,
                documents=documents,
                top_n=self._top_n or len(documents),
                max_tokens_per_doc=self._max_tokens_per_document,
                request_options=self._request_options,
            )
            print(response)
        return self._combine_reranked_results(response,rerankables)
