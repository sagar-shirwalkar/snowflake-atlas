# embedding-strategies — templates and worked examples

## Model Comparison Table (2026)

| Model                      | Dimensions | Max Tokens | Best For                            |
| -------------------------- | ---------- | ---------- | ----------------------------------- |
| **voyage-3-large**         | 1024       | 32000      | Claude apps (Anthropic recommended) |
| **voyage-3**               | 1024       | 32000      | Claude apps, cost-effective         |
| **voyage-code-3**          | 1024       | 32000      | Code search                         |
| **voyage-finance-2**       | 1024       | 32000      | Financial documents                 |
| **voyage-law-2**           | 1024       | 32000      | Legal documents                     |
| **text-embedding-3-large** | 3072       | 8191       | OpenAI apps, high accuracy          |
| **text-embedding-3-small** | 1536       | 8191       | OpenAI apps, cost-effective         |
| **bge-large-en-v1.5**      | 1024       | 512        | Open source, local deployment       |
| **all-MiniLM-L6-v2**       | 384        | 256        | Fast, lightweight                   |
| **multilingual-e5-large**  | 1024       | 512        | Multi-language                      |

## Embedding Pipeline

```
Document → Chunking → Preprocessing → Embedding Model → Vector
                ↓
        [Overlap, Size]  [Clean, Normalize]  [API/Local]
```

## Templates

### Template 1: Voyage AI Embeddings (Recommended for Claude)

```python
from langchain_voyageai import VoyageAIEmbeddings
from typing import List
import os

embeddings = VoyageAIEmbeddings(
    model="voyage-3-large",
    voyage_api_key=os.environ.get("VOYAGE_API_KEY")
)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    return embeddings.embed_documents(texts)

def get_query_embedding(query: str) -> List[float]:
    return embeddings.embed_query(query)

# Specialized models
code_embeddings = VoyageAIEmbeddings(model="voyage-code-3")
finance_embeddings = VoyageAIEmbeddings(model="voyage-finance-2")
legal_embeddings = VoyageAIEmbeddings(model="voyage-law-2")
```

### Template 2: OpenAI Embeddings

```python
from openai import OpenAI
from typing import List
import numpy as np

client = OpenAI()

def get_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    dimensions: int = None
) -> List[List[float]]:
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        kwargs = {"input": batch, "model": model}
        if dimensions:
            kwargs["dimensions"] = dimensions
        response = client.embeddings.create(**kwargs)
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
    return all_embeddings

def get_reduced_embedding(text: str, dimensions: int = 512) -> List[float]:
    """Matryoshka dimensionality reduction."""
    return get_embeddings([text], model="text-embedding-3-small", dimensions=dimensions)[0]
```

### Template 3: Local Embeddings with Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np

class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=normalize, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed([query])[0]
```

### Template 4: Chunking Strategies

```python
from typing import List, Tuple
import re

def chunk_by_tokens(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start = end - chunk_overlap
    return chunks

def chunk_by_semantic_sections(text: str, headers_pattern: str = r'^#{1,3}\s+.+$') -> List[Tuple[str, str]]:
    """Chunk markdown by headers, preserving hierarchy."""
    lines = text.split('\n')
    chunks = []
    current_header = ""
    current_content = []
    for line in lines:
        if re.match(headers_pattern, line, re.MULTILINE):
            if current_content:
                chunks.append((current_header, '\n'.join(current_content)))
            current_header = line
            current_content = []
        else:
            current_content.append(line)
    if current_content:
        chunks.append((current_header, '\n'.join(current_content)))
    return chunks
```

### Template 5: Embedding Quality Evaluation

```python
import numpy as np
from typing import List, Dict

def evaluate_retrieval_quality(
    queries: List[str],
    relevant_docs: List[List[str]],
    retrieved_docs: List[List[str]],
    k: int = 10
) -> Dict[str, float]:
    def precision_at_k(relevant: set, retrieved: List[str], k: int) -> float:
        retrieved_k = retrieved[:k]
        return len(set(retrieved_k) & relevant) / k if k > 0 else 0

    def recall_at_k(relevant: set, retrieved: List[str], k: int) -> float:
        retrieved_k = retrieved[:k]
        return len(set(retrieved_k) & relevant) / len(relevant) if relevant else 0

    def mrr(relevant: set, retrieved: List[str]) -> float:
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1 / (i + 1)
        return 0

    metrics = {f"precision@{k}": [], f"recall@{k}": [], "mrr": []}
    for relevant, retrieved in zip(relevant_docs, retrieved_docs):
        relevant_set = set(relevant)
        metrics[f"precision@{k}"].append(precision_at_k(relevant_set, retrieved, k))
        metrics[f"recall@{k}"].append(recall_at_k(relevant_set, retrieved, k))
        metrics["mrr"].append(mrr(relevant_set, retrieved))
    return {name: np.mean(values) for name, values in metrics.items()}
```
