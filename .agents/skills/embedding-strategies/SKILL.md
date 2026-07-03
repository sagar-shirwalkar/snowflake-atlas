---
name: embedding-strategies
description: Select and optimize embedding models for semantic search and RAG applications. Use when choosing embedding models, implementing chunking strategies, or optimizing embedding quality for specific domains. Branches: compare-models (evaluate candidates), optimize-chunking (tune chunk strategy), fine-tune-embeddings (domain adaptation).
disable-model-invocation: false
---

Guide to selecting, comparing, and optimizing embedding models for vector search and RAG.

## Leading words

- **Compare** — Evaluate embedding candidates against the project's retrieval criteria (latency, dimension, domain fit). Not every model suits every workload.
- **Chunk** — Split documents into embeddable pieces while preserving semantic boundaries. The chunking strategy is as important as the model.
- **Optimize** — Tune the pipeline: normalize embeddings, batch requests, cache results, reduce dimensions when possible.

## Phases

### PHASE 1: Scout

**Completion criterion:** a shortlist of 2-3 embedding models that match the project's constraints (language domain, latency budget, local vs API, dimension limits).

1. Identify the project's constraints: language, deployment environment (local/API), latency requirements, and cost budget.
2. Compare candidate models using the comparison table in `references/details.md`.
3. For local deployments, check ONNX/MLX compatibility and model size. For API deployments, check rate limits and pricing.
4. If the project uses Claude, consider Voyage AI models (Anthropic recommended).

### PHASE 2: Chunk

**Completion criterion:** a chunking strategy documented and implemented — chunk size, overlap, boundary detection, and preprocessing steps.

1. Choose a chunk size based on your embedding model's token limit (e.g., 512 tokens for `bge-base-en-v1.5`, 8191 for `text-embedding-3-large`).
2. Pick a boundary strategy:
   - **Semantic boundaries** (headings, paragraphs) — preserves meaning, uneven sizes.
   - **Fixed-size with overlap** — uniform chunks, may split mid-sentence.
3. Add metadata to each chunk (source, heading, position) for downstream filtering.
4. Preprocess text before embedding: normalize whitespace, strip boilerplate.

### PHASE 3: Optimize

**Completion criterion:** embedding pipeline integrated into the project's bundle build, with batch processing and optional caching.

1. Normalize all embeddings to unit length (required for cosine similarity).
2. Batch embed requests (e.g., 32-64 texts per batch) for throughput.
3. Cache embeddings for static content to avoid recomputing on rebuild.
4. If dimension reduction is needed (e.g., 3072 → 768 via PCA), evaluate the recall tradeoff before committing.

## Reference Files

| File | Contents |
|------|----------|
| `references/details.md` | Model comparison table, worked examples, code templates for Voyage AI, OpenAI, BGE, and local ONNX/MLX |
