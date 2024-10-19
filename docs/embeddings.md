# Embeddings

## Onnx Runtime

A convenient way to run onnx models to generate embeddings.

### With HuggingFace Model

The following example shows how to use the embedding function to download a model from HuggingFace and use it to generate embeddings.

> Note: You will need to install `huggingface_hub` (`pip install huggingface_hub`) to download the model.

```python
from chromadbx.embeddings.onnx import OnnxRuntimeEmbeddings
ef=OnnxRuntimeEmbeddings(model_path="snowflake/arctic-embed-s",preferred_providers=["CPUExecutionProvider"],hf_download=True)

ef(["text"])
```

### With Local Model

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

## Google Vertex AI

A convenient way to run Google Vertex AI models to generate embeddings.

Google Vertex AI uses variety of authentication methods. The most secure is either service account key file or Google Application Default Credentials.

```py
import chromadb
from chromadbx.embeddings.google import GoogleVertexAiEmbeddings

ef = GoogleVertexAiEmbeddings()

client = chromadb.Client()

col = client.get_or_create_collection("test", embedding_function=ef)

col.add(ids=["id1", "id2", "id3"], documents=["lorem ipsum...", "doc2", "doc3"])
```

### Auth with service account key file

```py
from chromadbx.embeddings.google import GoogleVertexAiEmbeddings
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file("path/to/service-account-key.json")
ef = GoogleVertexAiEmbeddings(credentials=credentials)

ef(["hello world", "goodbye world"])
```

## Mistral AI

A convenient way to generate embeddings using Mistral AI models.


```py
import chromadb
from chromadbx.embeddings.mistral import MistralAiEmbeddings

ef = MistralAiEmbeddings()

client = chromadb.Client()

col = client.get_or_create_collection("test", embedding_function=ef)

col.add(ids=["id1", "id2", "id3"], documents=["lorem ipsum...", "doc2", "doc3"])
col.query(query_texts=["lorem ipsum..."], n_results=2)
```
