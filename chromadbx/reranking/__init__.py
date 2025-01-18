from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
from typing_extensions import TypedDict  # TODO add typing_extensions to dependencies

from chromadb.api.types import (
    IDs,
    Documents,
    QueryResult,
    Document,
    PyEmbeddings,
    Embeddings,
    URI,
    Loadable,
    Metadata,
    Include,
)

Distance = float
Distances = List[Distance]
Rerankable = Union[Documents, QueryResult]

RerankerID = str
Queries = Union[str, List[str]]


class RerankedQueryResult(TypedDict):
    ids: List[IDs]
    embeddings: Optional[
        Union[
            List[Embeddings],
            List[PyEmbeddings],
            List[np.typing.NDArray[Union[np.int32, np.float32]]],
        ]
    ]
    documents: Optional[List[List[Document]]]
    uris: Optional[List[List[URI]]]
    data: Optional[List[Loadable]]
    metadatas: Optional[List[List[Metadata]]]
    distances: Optional[List[List[Distance]]]
    included: Include
    ranked_distances: Dict[RerankerID, List[Distance]]


class RerankedDocuments(TypedDict):
    documents: List[Documents]
    ranked_distances: Dict[RerankerID, Distances]


RankedResults = Union[List[Documents], List[RerankedQueryResult]]

D = TypeVar("D", bound=Rerankable, contravariant=True)
T = TypeVar("T", bound=RankedResults, covariant=True)


def validate_rerankables(queries: Queries, rerankables: D) -> None:
    """
    Validate the rerankables. Throws an exception if the rerankables are not valid.
    """
    if queries is None or len(queries) == 0:
        raise ValueError("Queries must not be empty")
    if rerankables is None or len(rerankables) == 0:
        raise ValueError("Rerankables results must not be empty")
    if isinstance(queries, list) and (
        isinstance(rerankables, list) and all(isinstance(s, str) for s in rerankables)
    ):
        raise ValueError("You can only rerank a single query at a time with Documents")
    if isinstance(rerankables, list) and not all(
        isinstance(s, str) for s in rerankables
    ):
        raise ValueError("Documents must be a list of Document (str)")
    elif isinstance(rerankables, dict):
        if (
            rerankables.get("documents", None) is None
            or len(rerankables["documents"]) == 0
        ):
            raise ValueError("QueryResult must have documents to rerank")
        if len(queries) != len(rerankables["documents"]):
            raise ValueError("Number of documents and queries must be the same")


@runtime_checkable
class RerankingFunction(Protocol[D, T]):
    """
    A function that reranks the results of a query.
    """

    def id(self) -> RerankerID:
        ...

    def __call__(self, queries: Queries, rerankables: D) -> T:
        ...

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        call = getattr(cls, "__call__")

        def __call__(
            self: RerankingFunction[D, T], queries: Queries, rerankables: D
        ) -> T:
            validate_rerankables(queries, rerankables)
            return cast(T, call(self, queries, rerankables))

        setattr(cls, "__call__", __call__)
