# 🧠 RAG Knowledge Base – Embeddings, Vector Store & Visualization

This project builds a Retrieval-Augmented Generation (RAG) foundation by transforming a Markdown knowledge base into vector embeddings, storing them in a vector database, and visualizing the semantic space to analyze and compare embedding models.

## 🔁 RAG Architecture Overview

This script implements the indexing phase of a RAG pipeline:
```
Knowledge Base (Markdown)
        ↓
Text Loading (LangChain)
        ↓
Text Splitting (Chunking)
        ↓
Embedding Encoding (BERT-based models)
        ↓
Vector Store (FAISS / Chroma)
        ↓
Retrieval + LLM Generation (future step)
```

The generated vector store can later be used by an LLM to retrieve relevant context before generating answers, improving accuracy and reducing hallucinations.

## 📚 Knowledge Base Ingestion (LangChain)

- Markdown files are loaded from `knowledge-base/**`
- LangChain document loaders are used to standardize ingestion
- Each document is enriched with metadata (`doc_type`) derived from its folder
- This metadata is preserved in the vector store for filtering and analysis

## ✂️ Text Splitting Strategy

Before embedding, documents are divided into chunks using:

- **RecursiveCharacterTextSplitter**
- **Chunk size:** 1000 characters
- **Overlap:** 200 characters

**Why splitting matters in RAG:**
- Prevents context window overflow
- Preserves semantic coherence
- Improves retrieval granularity
- Allows fine-grained similarity search

## 🔐 Token & Cost Awareness

The entire knowledge base is tokenized using `tiktoken`:

- Token count is calculated before embedding
- This allows:
  - Cost estimation
  - Model selection optimization
  - RAG pipeline scaling decisions

## 🧬 Embedding Models (Encoders)

The project uses sentence-level encoder models to transform text into dense vectors.

**Current implementation:**
- BERT-based encoder
- `all-MiniLM-L6-v2` (HuggingFace)
- Fast, low-cost, strong semantic performance

**Optional alternatives:**
- OpenAI embedding models (commented)
- Any BERT / SentenceTransformer model
- Allows side-by-side comparison of embedding spaces

Each chunk is encoded into a fixed-dimension vector representing its semantic meaning.

## 🗄 Vector Store (FAISS / Chroma)

Embeddings are stored in a vector database:

- This project uses **Chroma** (FAISS-compatible architecture)
- Persistent on disk (`vector_db/`)

**Why a vector DB in RAG?**
- Enables fast similarity search (cosine / L2)
- Scales to thousands of documents
- Supports metadata filtering
- Required for real-time retrieval by LLMs

## 🔍 Embedding Space Analysis

After indexing, all vectors are extracted to analyze:

- Number of vectors
- Embedding dimensionality
- Distribution across document types

This step is critical to validate embedding quality before plugging into a RAG agent.

## 📊 Visualization with t-SNE

To make embeddings interpretable by humans:

**Dimensionality Reduction:**
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding)
- Reduces vectors:
  - From high dimensions → 2D
  - From high dimensions → 3D

**Visual Output:**
- Interactive Plotly charts
- Each point = one text chunk
- Color-coded by document type
- Hover shows text preview

## 🧪 Why Visualization Matters

- Detect semantic clusters
- Identify overlaps between document types
- Spot noisy or poorly embedded content
- Compare different embedding models visually
- Validate chunking and encoder choices before production

## 🔄 Comparing Embedding Models

Because the pipeline is modular, you can:

- Swap encoder models (BERT variants, OpenAI, etc.)
- Rebuild the vector store
- Visualize and compare:
  - Cluster separation
  - Semantic density
  - Retrieval relevance

This makes the project ideal for embedding benchmarking in RAG systems.

## 🎯 Use Cases

- RAG chatbots
- Internal company knowledge assistants
- Contract / HR / product documentation search
- AI agents powered by LangChain
- Embedding model evaluation & research

## 🛠 Tech Stack

- **LangChain** – document loading, splitting, vector orchestration
- **BERT / Sentence Transformers** – semantic encoders
- **FAISS / Chroma** – vector storage & retrieval
- **t-SNE (scikit-learn)** – dimensionality reduction
- **Plotly** – interactive visualization
- **Python** – orchestration

## 📦 Outputs

- Persistent vector database (`vector_db/`)
- Interactive 2D & 3D embedding maps
- A reusable RAG-ready index
