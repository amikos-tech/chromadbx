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

> [!NOTE]
> It is our intent that all officially supported reranking functions shall return distances instead of scores to be consistent with the core Chroma project. However, this is not a hard requirement and you should check the documentation for each reranking function you plan to use.

The following reranking functions are supported:

| Reranking Function | Official Docs |
| ------------------ | ------------- |
| [Cohere](#cohere) | [docs](https://docs.cohere.com/docs/rerank-2) |

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

collection = client.get_collection("documents")

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
