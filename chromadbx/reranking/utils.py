from typing import List, Tuple

from chromadbx.reranking import Queries, Rerankable


def get_query_documents_tuples(
    queries: Queries, rerankables: Rerankable
) -> List[Tuple[str, List[str]]]:
    if (
        isinstance(queries, str)
        and isinstance(rerankables, list)
        and all(isinstance(d, str) for d in rerankables)
    ):
        return [(queries, rerankables)]
    elif (
        isinstance(queries, list)
        and isinstance(rerankables, dict)
        and "documents" in rerankables
        and len(rerankables["documents"]) == len(queries)
    ):
        return [
            (query, document)
            for query, document in zip(queries, rerankables["documents"])
        ]
    else:
        raise ValueError("Invalid input types")
