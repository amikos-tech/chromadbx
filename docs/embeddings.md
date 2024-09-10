# Embeddings

## Onnx Runtime

A convenient way to run onnx models to generate embeddings.

Download a model:

```bash
mkdir -p ~/.cache/models/hf
git clone https://huggingface.co/snowflake/arctic-embed-s ~/.cache/models/hf/snowflake-arctic-embed-s
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

## Llama.cpp

⚠️ Llama.cpp embedding function is still in early development. Please report any problems you may have by raising an
issue.

A convenient way to run llama.cpp models to generate embeddings.

### With HF Model

The embedding function supports downloading models directly from HuggingFace Hub.

```py
import chromadb
from chromadbx.embeddings.llamacpp import LlamaCppEmbeddingFunction

ef = LlamaCppEmbeddingFunction(model_path="yixuan-chia/snowflake-arctic-embed-s-GGUF",
                               hf_file_name="snowflake-arctic-embed-s-F32.gguf")

client = chromadb.Client()

col = client.get_or_create_collection("test", embedding_function=ef)

col.add(ids=["id1", "id2", "id3"], documents=["lorem ipsum...", "doc2", "doc3"])

```

### With Local Model

Download a model:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download ChristianAzinn/snowflake-arctic-embed-s-gguf --include=snowflake-arctic-embed-s-f16.GGUF --local-dir snowflake-arctic-embed-s
```

```python
import chromadb
from chromadbx.embeddings.llamacpp import LlamaCppEmbeddingFunction

ef = LlamaCppEmbeddingFunction(model_path="snowflake-arctic-embed-s/snowflake-arctic-embed-s-f16.GGUF")

client = chromadb.Client()

col = client.get_or_create_collection("test", embedding_function=ef)

col.add(ids=["id1", "id2", "id3"], documents=["lorem ipsum...", "doc2", "doc3"])
```
