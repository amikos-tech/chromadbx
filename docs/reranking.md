# Reranking

Reranking is a process of reordering a list of items based on their relevance to a query. This project supports reranking of documents and query results.

```python
from chromadbx.reranking.some_reranker import SomeReranker
import chromadb
some_reranker = SomeReranker()

client = chromadb.Client()

collection = client.get_collection("documents")

results = collection.query(
    query_texts=["What is the capital of the United States?"],
    n_results=10,
)

reranked_results = some_reranker(results)

print("Documents:", reranked_results["documents"][0])
print("Distances:", reranked_results["distances"][0])
print("Reranked distances:", reranked_results["ranked_distances"][some_reranker.id()][0])
```

Reranking results take the following two forms.

**Reranked documents**

This is intended for more simple use cases and also to be used as standalone feature.

```python
class RerankedDocuments(TypedDict):
    documents: List[Documents]
    ranked_distances: Dict[RerankerID, Distances]
```

**Reranked Chroma results**

This type of reranking results are indended to be used together with Chroma and more specifically to be a drop-in replacement for existing Chroma `QueryResult` type.

```python
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
    ranked_distances: Dict[RerankerID, List[Distances]]

```

**Common types**

```python
RerankerID = str # the ID of the reranker - this is used to identify the reranker in the reranked results
Distance = float # the distance between the query and the document
Distances = List[Distance] # a list of distances

Rerankable = Union[Documents, QueryResult] # a type that can be reranked
Queries = Union[str, List[str]] # a type that can be used as a query
```

> [!NOTE]
> It is our intent that all supported reranking functions shall return distances instead of scores to be consistent with the core Chroma project. However, this is not a hard requirement and you should check the documentation for each reranking function you plan to use.

The following reranking functions are supported:

| Reranking Function | Official Docs |
| ------------------ | ------------- |
| [Cohere](#cohere) | [docs](https://docs.cohere.com/docs/rerank-2) |
| [Together](#together) | [docs](https://docs.together.ai/docs/rerank-overview) |

## Cohere

Cohere reranking function offers a convinient wrapper around the Cohere API to rerank documents and query results. For more information on Cohere reranking, visit the official [docs](https://docs.cohere.com/docs/rerank-2) or [API docs](https://docs.cohere.com/reference/rerank).

You need to install the `cohere` package to use this reranking function.


```bash
pip install cohere # or poetry add cohere
```

Before using the reranking function, you need to obtain [Cohere API](https://dashboard.cohere.com/api-keys) key and set the `COHERE_API_KEY` environment variable.

> [!TIP]
>  By default, the reranking function will return distances. If you need to get the raw scores, set the `raw_scores` parameter to `True`.

```python
import os
import chromadb
from chromadbx.reranking import CohereReranker

cohere = CohereReranker(api_key=os.getenv("COHERE_API_KEY"))

client = chromadb.Client()

collection = client.get_or_create_collection("capital_cities")

collection.add(
    ids=["usa", "france", "germany", "italy", "spain"],
    documents=[
        "The capital of the United States is Washington, D.C.",
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome.",
        "The capital of Spain is Madrid.",
    ]
)

results = collection.query(
    query_texts=["What is the capital of the United States?"],
    n_results=10,
)

reranked_results = cohere(results)
```

Available options:

- `api_key`: The Cohere API key.
- `model_name`: The Cohere model to use for reranking. Defaults to `rerank-v3.5`.
- `raw_scores`: Whether to return the raw scores from the Cohere API. Defaults to `False`.
- `top_n`: The number of results to return. Defaults to `None`.
- `max_tokens_per_document`: The maximum number of tokens per document. Defaults to `4096`.
- `timeout`: The timeout for the Cohere API request. Defaults to `60`.
- `max_retries`: The maximum number of retries for the Cohere API request. Defaults to `3`.
- `additional_headers`: Additional headers to include in the Cohere API request. Defaults to `None`.

## Together

Together reranking function offers a convinient wrapper around the Together API to rerank documents and query results. For more information on Together reranking, visit the [official docs](https://docs.together.ai/docs/rerank-overview) or [API docs](https://docs.together.ai/reference/rerank-1).

You need to install the `together` package to use this reranking function.

```bash
pip install --upgrade together # or poetry add together
```

Before using the reranking function, you need to obtain [Together API](https://api.together.xyz/settings/api-keys) key and set the `TOGETHER_API_KEY` environment variable.

> [!TIP]
>  By default, the reranking function will return distances. If you need to get the raw scores, set the `raw_scores` parameter to `True`.

```python
import os
import chromadb
from chromadbx.reranking import TogetherReranker

together = TogetherReranker(api_key=os.getenv("TOGETHER_API_KEY"))

client = chromadb.Client()

collection = client.get_or_create_collection("capital_cities")

collection.add(
    ids=["usa", "france", "germany", "italy", "spain"],
    documents=[
        "The capital of the United States is Washington, D.C.",
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome.",
        "The capital of Spain is Madrid.",
    ]
)

results = collection.query(
    query_texts=["What is the capital of the United States?"],
    n_results=10,
)

reranked_results = together(results)
```

Available options:

- `api_key`: The Together API key.
- `model_name`: The Together model to use for reranking. Defaults to `Salesforce/Llama-Rank-V1`.
- `raw_scores`: Whether to return the raw scores from the Together API. Defaults to `False`.
- `top_n`: The number of results to return. Defaults to `None`.
- `timeout`: The timeout for the Together API request. Defaults to `60`.
- `max_retries`: The maximum number of retries for the Together API request. Defaults to `3`.
- `additional_headers`: Additional headers to include in the Together API request. Defaults to `None`.
