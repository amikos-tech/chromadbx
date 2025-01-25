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


class TogetherReranker(RerankingFunction[Rerankable, RankedResults]):
    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = "Salesforce/Llama-Rank-V1",
        *,
        raw_scores: bool = False,
        top_n: Optional[int] = None,
        timeout: Optional[int] = 60,
        max_retries: Optional[int] = 3,
        additional_headers: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the TogetherReranker. Information on available models can be found [here](https://docs.together.ai/docs/serverless-models#rerank-models)

        Args:
            api_key: The Together API key.
            model_name: The Together model to use for reranking. Defaults to `Salesforce/Llama-Rank-V1`.
            raw_scores: Whether to return the raw scores from the Together API. Defaults to `False`.
            top_n: The number of results to return. Defaults to `None`.
            timeout: The timeout for the Together API request. Defaults to `60`.
            max_retries: The maximum number of retries for the Together API request. Defaults to `3`.
            additional_headers: Additional headers to include in the Together API request. Defaults to `None`.
        """
        try:
            import together
        except ImportError:
            raise ValueError(
                "The together python package is not installed. Please install it with `pip install --upgrade together`"
            )
        if not api_key and not os.getenv("TOGETHER_API_KEY"):
            raise ValueError(
                "API key is required. Please set the TOGETHER_API_KEY environment variable or pass it directly."
            )
        if not model_name:
            raise ValueError(
                "Model name is required. Please set the model_name parameter or use the default value."
            )
        self._client = together.Together(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            supplied_headers=additional_headers,
        )
        self._model_name = model_name
        self._top_n = top_n
        self._raw_scores = raw_scores

    def id(self) -> RerankerID:
        return RerankerID("together")

    def _combine_reranked_results(
        self, results_list: List["together.types.RerankResponse"], rerankables: Rerankable  # type: ignore # noqa: F821
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
        """
        Get the reranked results for a list of queries and documents.

        Args:
            queries (Queries): A list of queries to rerank.
            rerankables (Rerankable): A list of documents to rerank.

        Returns:
            RankedResults: The reranked results.

        Example:
            >>> from chromadbx.reranking.together import TogetherReranker
            >>> reranker = TogetherReranker(api_key="your_api_key")
            >>> queries = ["What is the capital of France?"]
            >>> documents = ["Washington is the capital of the United States.", "Paris is the capital of France.", "Berlin is the capital of Germany."]
            >>> reranked_results = reranker(queries, documents)
        """
        query_documents_tuples = get_query_documents_tuples(queries, rerankables)
        results = []
        for query, documents in query_documents_tuples:
            response = self._client.rerank.create(
                model=self._model_name,
                query=query,
                documents=documents,
                top_n=self._top_n or len(documents),
            )
            results.append(response)
        return self._combine_reranked_results(results, rerankables)
