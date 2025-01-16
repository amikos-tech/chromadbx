from typing import Dict, List, Protocol, TypeVar, Union, cast, runtime_checkable

from typing_extensions import TypedDict  # TODO add typing_extensions to dependencies

from chromadb.api.types import Documents, QueryResult, Document

Score = float
Scores = List[Score]
Rerankable = Union[Documents, QueryResult]

RerankerID = str
Queries = Union[str, List[str]]


class RerankedQueryResult(QueryResult):  # type: ignore
    scores: Dict[RerankerID, List[Scores]]


class RerankedDocuments(TypedDict):
    documents: List[Documents]
    scores: Dict[RerankerID, Scores]


RankedResults = Union[List[Documents], List[RerankedQueryResult]]

D = TypeVar("D", bound=Rerankable, contravariant=True)
T = TypeVar("T", bound=RankedResults, covariant=True)


def validate_rerankables(queries: Queries, rerankables: D) -> None:
    """
    Validate the rerankables. Throws an exception if the rerankables are not valid.
    """

    if queries is None or not isinstance(queries, str) or not isinstance(queries, list):
        raise ValueError("Queries must be a string or a list of strings")
    if rerankables is None or len(rerankables) == 0:
        raise ValueError("Rerankables results must not be empty")
    if isinstance(queries, list) and isinstance(rerankables, Documents):
        raise ValueError("You can only rerank a single query at a time with Documents")
    if isinstance(rerankables, Documents) and not all(
        isinstance(document, Document) for document in rerankables
    ):
        raise ValueError("Documents must be a list of Document (str)")
    elif isinstance(rerankables, QueryResult):
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
        # Raise an exception if __call__ is not defined since it is expected to be defined
        call = getattr(cls, "__call__")

        def __call__(
            self: RerankingFunction[D, T], queries: Queries, rerankables: D
        ) -> T:
            validate_rerankables(queries, rerankables)
            return cast(T, call(self, rerankables))

        setattr(cls, "__call__", __call__)
