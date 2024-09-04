# Embeddings

## Onnx Runtime

A convenient way to run onnx models to generate embeddings.

Download a model:

```bash
mkdir -p ~/.cache/models/hf
git clone git clone https://huggingface.co/snowflake/arctic-embed-s ~/.cache/models/hf/snowflake-arctic-embed-s
```

Alternatively use HF CLI:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download snowflake/arctic-embed-s --local-dir snowflake-arctic-embed-s
```

```python
from chromadbx.embeddings.onnx import OnnxRuntimeEmbeddings
import chromadb

# adjust the model path to the path where you downloaded the model
# We use the CPUExecutionProvider, adjust it to your liking or leave empty to let onnx choose the most appropriate provider
ef = OnnxRuntimeEmbeddings(model_path=f"snowflake-arctic-embed-s", preferred_providers=["CPUExecutionProvider"])

client = chromadb.Client()

col = client.get_or_create_collection("test", embedding_function=ef)

col.add(ids=["id1", "id2", "id3"], documents=["lorem ipsum...", "doc2", "doc3"])
```