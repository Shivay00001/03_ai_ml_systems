# 03_ai_ml_systems - RAG Pipeline

> Production-grade Retrieval Augmented Generation (RAG) system demonstrating modern LLM application architecture.

## 🎯 Overview

This module implements a comprehensive RAG pipeline:

- **Document Processing** - PDF, Markdown, Text ingestion
- **Vector Store** - ChromaDB for embeddings
- **LLM Integration** - OpenAI/Anthropic adapters
- **Retrieval** - Semantic search with reranking
- **Generation** - Context-aware response generation

## 📁 Structure

```
03_ai_ml_systems/
├── src/
│   ├── core/                # Core RAG components
│   │   ├── embeddings.py    # Embedding models
│   │   ├── vector_store.py  # Vector database
│   │   └── llm.py           # LLM clients
│   ├── ingestion/           # Document processing
│   │   ├── loaders.py       # Document loaders
│   │   ├── chunkers.py      # Text chunking
│   │   └── pipeline.py      # Ingestion pipeline
│   ├── retrieval/           # Retrieval logic
│   │   ├── retriever.py     # Semantic search
│   │   └── reranker.py      # Result reranking
│   └── generation/          # Response generation
│       ├── chain.py         # RAG chain
│       └── prompts.py       # Prompt templates
├── tests/                   # Test suite
└── example_data/            # Sample documents
```

## 🚀 Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -e .

# Set environment variables
export OPENAI_API_KEY=your-api-key

# Run example
python -m src.main
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                       │
│        Documents → Loaders → Chunkers → Embeddings          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      VECTOR STORE                           │
│              ChromaDB / FAISS / Pinecone                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   RETRIEVAL & RANKING                       │
│          Semantic Search → Reranking → Context              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                           │
│           Prompt Template + Context → Response              │
└─────────────────────────────────────────────────────────────┘
```

## 📄 License

MIT
