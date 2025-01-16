from typing import Dict, List, Protocol, TypeVar, Union, runtime_checkable

from typing_extensions import TypedDict # TODO add typing_extensions to dependencies

from chromadb.api.types import Documents, QueryResult, Document
Score = float
Scores = List[Score]
Rerankable = Union[Documents, QueryResult]

RerankerID = str
class RerankedQueryResult(QueryResult):
    scores: Dict[RerankerID, List[Scores]]

class RerankedDocuments(TypedDict):
    documents: List[Documents]
    scores: Dict[RerankerID, Scores]

RankedResults = Union[List[Documents], List[RerankedQueryResult]]

D = TypeVar("D", bound=Rerankable, contravariant=True) # type: ignore
T = TypeVar("T", bound=RankedResults, contravariant=True) # type: ignore


def validate_rerankables(rerankables: D):
    """
    Validate the rerankables. Throws an exception if the rerankables are not valid.
    """
    
    if isinstance(rerankables, Documents):
        if all(isinstance(document, Document) for document in rerankables):
            return
        raise ValueError("Documents must be a list of Document")
    elif isinstance(rerankables, QueryResult):
        if rerankables.get("documents", None) is None or len(rerankables["documents"]) == 0:
            raise ValueError("QueryResult must have documents to rerank")
        

@runtime_checkable
class RerankingFunction(Protocol[D,T]):
    """
    A function that reranks the results of a query.
    """

    def id(self) -> RerankerID:
        ...

    def __call__(self, rerankables: D) -> T:
        ...

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # Raise an exception if __call__ is not defined since it is expected to be defined
        call = getattr(cls, "__call__")

        def __call__(self: RerankingFunction[D], rerankables: D) -> T:
            validate_rerankables(rerankables)
            return call(self, rerankables)
        
        setattr(cls, "__call__", __call__)
