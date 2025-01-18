import os
from typing import Any, Dict, Optional, List

from chromadbx.reranking import (
    Queries,
    RankedResults,
    Rerankable,
    RerankedDocuments,
    RerankedQueryResult,
    RerankerID,
    RerankingFunction,
)
from chromadbx.reranking.utils import get_query_documents_tuples


class CohereReranker(RerankingFunction[Rerankable, RankedResults]):
    def __init__(
        self,
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
        """
        Initialize the CohereReranker.

        Args:
            api_key: The Cohere API key.
            model_name: The Cohere model to use for reranking. Defaults to `rerank-v3.5`.
            raw_scores: Whether to return the raw scores from the Cohere API. Defaults to `False`.
            top_n: The number of results to return. Defaults to `None`.
            max_tokens_per_document: The maximum number of tokens per document. Defaults to `4096`.
            timeout: The timeout for the Cohere API request. Defaults to `60`.
            max_retries: The maximum number of retries for the Cohere API request. Defaults to `3`.
            additional_headers: Additional headers to include in the Cohere API request. Defaults to `None`.
        """
        try:
            import cohere
            from cohere.core.request_options import RequestOptions
        except ImportError:
            raise ImportError(
                "cohere is not installed. Please install it with `pip install cohere`"
            )
        if not api_key and not os.getenv("COHERE_API_KEY"):
            raise ValueError(
                "API key is required. Please set the COHERE_API_KEY environment variable or pass it directly."
            )
        if not model_name:
            raise ValueError(
                "Model name is required. Please set the model_name parameter or use the default value."
            )
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

    def _combine_reranked_results(
        self, results_list: List["cohere.v2.types.V2RerankResponse"], rerankables: Rerankable  # type: ignore # noqa: F821
    ) -> RankedResults:
        all_ordered_scores = []

        for results in results_list:
            if self._raw_scores:
                ordered_scores = [
                    r.relevance_score
                    for r in sorted(results.results, key=lambda x: x.index)  # type: ignore
                ]
            else:  # by default we calculate the distance to make results comparable with Chroma distance
                ordered_scores = [
                    1 - r.relevance_score
                    for r in sorted(results.results, key=lambda x: x.index)  # type: ignore
                ]
            all_ordered_scores.append(ordered_scores)

        if isinstance(rerankables, dict):
            combined_ordered_scores = [
                score for sublist in all_ordered_scores for score in sublist
            ]
            if len(rerankables["ids"]) != len(combined_ordered_scores):
                combined_ordered_scores = combined_ordered_scores + [None] * (
                    len(rerankables["ids"]) - len(combined_ordered_scores)
                )
            return RerankedQueryResult(
                ids=rerankables["ids"],
                embeddings=rerankables["embeddings"]
                if "embeddings" in rerankables
                else None,
                documents=rerankables["documents"]
                if "documents" in rerankables
                else None,
                uris=rerankables["uris"] if "uris" in rerankables else None,
                data=rerankables["data"] if "data" in rerankables else None,
                metadatas=rerankables["metadatas"]
                if "metadatas" in rerankables
                else None,
                distances=rerankables["distances"]
                if "distances" in rerankables
                else None,
                included=rerankables["included"] if "included" in rerankables else None,
                ranked_distances={self.id(): combined_ordered_scores},
            )
        elif isinstance(rerankables, list):
            if len(results_list) > 1:
                raise ValueError("Cannot rerank documents with multiple results")
            combined_ordered_scores = [
                score for sublist in all_ordered_scores for score in sublist
            ]
            if len(rerankables) != len(combined_ordered_scores):
                combined_ordered_scores = combined_ordered_scores + [None] * (
                    len(rerankables) - len(combined_ordered_scores)
                )
            return RerankedDocuments(
                documents=rerankables,
                ranked_distances={self.id(): combined_ordered_scores},
            )
        else:
            raise ValueError("Invalid rerankables type")

    def __call__(self, queries: Queries, rerankables: Rerankable) -> RankedResults:
        query_documents_tuples = get_query_documents_tuples(queries, rerankables)
        results = []
        for query, documents in query_documents_tuples:
            response = self._client.rerank(
                model=self._model_name,
                query=query,
                documents=documents,
                top_n=self._top_n or len(documents),
                max_tokens_per_doc=self._max_tokens_per_document,
                request_options=self._request_options,
            )
            results.append(response)
        return self._combine_reranked_results(results, rerankables)
