# Local Graph RAG Techniques

A collection of focused Graph RAG experiments. **GRAPHRAG_1** is the first module: a chunk-similarity knowledge graph with local Ollama generation, FAISS retrieval, NLP-based concept extraction, and optional LangSmith tracing.

## GRAPHRAG_1: Chunk-Similarity Graph RAG

The runnable, annotated implementation is [`Notebooks/graph_rag_explained_ollama_langsmith.ipynb`](Notebooks/graph_rag_explained_ollama_langsmith.ipynb).

### Techniques used

1. **Document loading and chunking** – read PDF pages with source/page metadata, then split them into overlapping text chunks.
2. **Local vector retrieval** – embed chunks with Ollama's `nomic-embed-text` model and index them in FAISS.
3. **NLP and concept extraction** – spaCy identifies named entities; the local Qwen model extracts general concepts as JSON; NLTK tokenizes and lemmatizes concepts.
4. **Weighted knowledge graph** – NetworkX represents chunks as nodes. Edges connect chunks above a cosine-similarity threshold, with shared concepts contributing to edge weights.
5. **Graph-assisted answering** – FAISS selects starting chunks, a priority queue expands through stronger graph connections, and Qwen checks context sufficiency and generates the final answer.
6. **Explainability and tracing** – visualize the traversal and preserve document metadata. LangSmith tracing is opt-in and may send questions, prompts, retrieved context, and answers to LangSmith.

This is a graph over **similar text chunks**, not an explicit entity-and-relationship graph. The notebook documents that distinction and the algorithm's quadratic graph-building cost.

## Local setup (Windows)

Use the project's existing virtual environment from the repository root:

```powershell
& "D:\CUDA_ENV\CUDA_ENV\Scripts\Activate.ps1"
python -m pip install -r Notebooks\requirements-graph-rag-local.txt
python -m spacy download en_core_web_sm
```

Start Ollama and make sure both model roles are available:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
ollama list
```

`qwen2.5:3b` is the chat/concept-extraction model. `nomic-embed-text` is a separate embedding model used by FAISS; the chat model cannot replace it for embeddings. Configure alternate installed model names with `OLLAMA_CHAT_MODEL` and `OLLAMA_EMBEDDING_MODEL`.

Open the notebook and select `D:\CUDA_ENV\CUDA_ENV\Scripts\python.exe` as the kernel. It loads PDFs from `Notebooks\Documents`. That folder is intentionally excluded from Git; add your own documents locally. For an initial run, use `MAX_PAGES_PER_PDF = 10`; set it to `None` to load every page. The notebook batches embedding requests to keep them bounded, but full graph construction and per-chunk LLM extraction can still take substantial time and memory for large collections.

### Optional LangSmith tracing

Tracing is disabled by default. To enable it locally, set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in an untracked local environment file or process environment. Never commit API keys. Ollama inference stays local, but enabled LangSmith tracing sends trace contents to the LangSmith service; do not enable it for sensitive documents unless that transfer is acceptable.

## Repository layout

```text
Notebooks/
  graph_rag_explained_ollama_langsmith.ipynb
  requirements-graph-rag-local.txt
  Documents/                          # local-only PDFs; not tracked
Originals/                            # local reference notebooks; licensing review required
```

The `Originals/` reference notebooks and local PDFs are not included in this repository. The reference material has separate redistribution terms; review those terms before publishing or re-licensing it.

## Adding the next technique

Add each new technique as a clearly named notebook or module, with Markdown explaining its architecture, dependencies, configuration, and execution steps. Keep credentials and private source documents out of commits, update this README with the new module, and validate the notebook using the project virtual environment before pushing.
