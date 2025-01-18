# ChromaX: An experimental utilities package for Chroma AI application database

## Installation

```bash
pip install chromadbx
```

## Features

- [Query Builder](https://github.com/amikos-tech/chromadbx#queries) - build queries using a builder pattern
- [ID generation](https://github.com/amikos-tech/chromadbx#id-generation) - generate IDs for documents
- [Embeddings](https://github.com/amikos-tech/chromadbx/blob/main/docs/embeddings.md) - generate embeddings for your documents:
    - [OnnxRuntime](https://github.com/amikos-tech/chromadbx/blob/main/docs/embeddings.md#onnx-runtime) embeddings
    - [Llama.cpp](https://github.com/amikos-tech/chromadbx/blob/main/docs/embeddings.md#llamacpp) embeddings
    - [Google Vertex AI](https://github.com/amikos-tech/chromadbx/blob/main/docs/embeddings.md#google-vertex-ai) embeddings
    - [Mistral AI](https://github.com/amikos-tech/chromadbx/blob/main/docs/embeddings.md#mistral-ai) embeddings
    - [Cloudflare Workers AI](https://github.com/amikos-tech/chromadbx/blob/main/docs/embeddings.md#cloudflare-workers-ai) embeddings
    - [SpaCy](https://github.com/amikos-tech/chromadbx/blob/main/docs/embeddings.md#spacy) embeddings
    - [Together](https://github.com/amikos-tech/chromadbx/blob/main/docs/embeddings.md#together) embeddings.
    - [Nomic](https://github.com/amikos-tech/chromadbx/blob/main/docs/embeddings.md#nomic) embeddings.
- [Reranking](https://github.com/amikos-tech/chromadbx/blob/main/docs/reranking.md) - rerank documents and query results using Cohere, OpenAI, or custom reranking functions.
    - [Cohere](https://github.com/amikos-tech/chromadbx/blob/main/docs/reranking.md#cohere) - rerank documents and query results using Cohere.

## Usage

### Queries

Supported filters:

- `$eq` - equal to (string, int, float)
- `$ne` - not equal to (string, int, float)
- `$gt` - greater than (int, float)
- `$gte` - greater than or equal to (int, float)
- `$lt` - less than (int, float)
- `$lte` - less than or equal to (int, float)
- `$in` - in (list of strings, ints, floats,bools)
- `$nin` - not in (list of strings, ints, floats,bools)

**Where:**


```python
import chromadb

from chromadbx.core.queries import eq, where, ne, and_

client = chromadb.PersistentClient(path="path/to/db")
collection = client.get_collection("collection_name")
collection.query(where=where(and_(eq("a", 1), ne("b", "2"))))
# {'$and': [{'a': ['$eq', 1]}, {'b': ['$ne', '2']}]}
```

**Where Document:**

```python
import chromadb

from chromadbx.core.queries import where_document, contains, not_contains, LogicalOperator

client = chromadb.PersistentClient(path="path/to/db")
collection = client.get_collection("collection_name")
collection.query(where_document=where_document(contains("this is a document", "this is another document")))
# {'$and': [{'$contains': 'this is a document'}, {'$contains': 'this is another document'}]}
collection.query(
    where_document=where_document(contains("this is a document", "this is another document", op=LogicalOperator.OR)))
# {'$or': [{'$contains': 'this is a document'}, {'$contains': 'this is another document'}]}
```

### ID Generation

```python
import chromadb
from chromadbx import IDGenerator
from functools import partial
from typing import Generator

def sequential_generator(start: int = 0) -> Generator[str, None, None]:
        _next = start
        while True:
            yield f"{_next}"
            _next += 1
client = chromadb.Client()
col = client.get_or_create_collection("test")
my_docs = [f"Document {_}" for _ in range(10)]
idgen = IDGenerator(len(my_docs), generator=partial(sequential_generator, start=10))
col.add(ids=idgen, documents=my_docs)
```

#### UUIDs (default)

```python
import chromadb
from chromadbx import UUIDGenerator

client = chromadb.Client()
col = client.get_or_create_collection("test")
my_docs = [f"Document {_}" for _ in range(10)]
col.add(ids=UUIDGenerator(len(my_docs)), documents=my_docs)
```

#### ULIDs

```python
import chromadb
from chromadbx import ULIDGenerator
client = chromadb.Client()
col = client.get_or_create_collection("test")
my_docs = [f"Document {_}" for _ in range(10)]
col.add(ids=ULIDGenerator(len(my_docs)), documents=my_docs)
```

#### Hashes

**Random SHA256:**

```python
import chromadb
from chromadbx import RandomSHA256Generator
client = chromadb.Client()
col = client.get_or_create_collection("test")
my_docs = [f"Document {_}" for _ in range(10)]
col.add(ids=RandomSHA256Generator(len(my_docs)), documents=my_docs)
```

**Document-based SHA256:**

```python
import chromadb
from chromadbx import DocumentSHA256Generator
client = chromadb.Client()
col = client.get_or_create_collection("test")
my_docs = [f"Document {_}" for _ in range(10)]
col.add(ids=DocumentSHA256Generator(documents=my_docs), documents=my_docs)
```

#### NanoID

```python
import chromadb
from chromadbx import NanoIDGenerator
client = chromadb.Client()
col = client.get_or_create_collection("test")
my_docs = [f"Document {_}" for _ in range(10)]
col.add(ids=NanoIDGenerator(len(my_docs)), documents=my_docs)
```
