# ChromaX: An experimental utilities package for Chroma vector database

## Installation

```bash
pip install chromadbx
```

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

```python
import chromadb

from chromadbx.core.queries import eq, where, ne, and_

client = chromadb.PersistentClient(path="path/to/db")
collection = client.get_collection("collection_name")
collection.query(where=where(and_(eq("a", 1), ne("b", "2"))))
```
