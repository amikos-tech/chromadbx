# Embeddings

## Onnx Runtime

A convenient way to run onnx models to generate embeddings.

Download a model:

```bash
mkdir -p ~/.cache/models/hf
git clone git clone https://huggingface.co/snowflake/arctic-embed-s ~/.cache/models/hf/snowflake-arctic-embed-s
```

```python
from pathlib import Path
from chromadbx.embeddings.ort import OnnxRuntimeEmbeddings
import chromadb
ef=OnnxRuntimeEmbeddings(model_path=f"{Path.home()}/Downloads/hf_experiments/snowflake-arctic-embed-s",preferred_providers=["CPUExecutionProvider"])

client = chromadb.Client()

col = client.get_or_create_collection("test", embedding_function=ef)

col.add(ids=["id1", "id2", "id3"], documents=["lorem ipsum...", "doc2", "doc3"])

```