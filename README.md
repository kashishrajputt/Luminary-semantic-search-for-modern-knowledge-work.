# Luminary – Semantic Search Engine

Luminary is a lightweight, production‑ready semantic search system designed for teams that work with large collections of text documents. It focuses on fast indexing, accurate semantic retrieval, and a clean workflow that can be deployed locally or in the cloud.

This project combines transformer‑based embeddings, FAISS vector search, a caching layer, a REST API, and a Streamlit UI into a single cohesive pipeline.

---

## Overview

Luminary takes plain text documents, cleans them, converts them into dense embeddings, and stores them efficiently so that repeated runs do not waste time recomputing anything.  
It uses sentence‑transformers for embedding generation and FAISS for fast similarity search.

The project includes:

- A document preprocessor  
- A smart SQLite‑based cache  
- An embedding generator  
- A FAISS search engine  
- A FastAPI server  
- A Streamlit UI  
- A CLI for indexing, searching, and diagnostics  

The aim is to keep everything understandable, maintainable, and simple to run.

---

## How Caching Works

The caching system avoids recomputing embeddings every time you run indexing.  
Each document is assigned a **SHA256 hash** based on its content.

The process:

1. When indexing begins, Luminary reads every document.  
2. It computes a SHA256 hash for the content.  
3. The cache is checked:  
   - If an entry with the same hash exists → the stored embedding is reused.  
   - If the file changed or is new → a fresh embedding is computed.  
4. Updated embeddings are written back into the SQLite database.

This keeps indexing fast and ensures correctness even when files change.

Typical numbers:
- First run (201 docs): around 7 seconds  
- Subsequent runs: under 1 second (100% cache hit rate)  

---

## How to Generate Embeddings (Indexing)

Indexing can be triggered through the CLI:

```bash
python main.py index --data-dir data/
```

This performs:
- Document preprocessing
- Cache lookup
- Embedding generation (only for changed documents)
- FAISS index creation
- Persistent index saving inside the `cache/` folder

Make sure your `.txt` files are inside the `data/` directory before running this.

---

## How to Start the API

Luminary provides a FastAPI server that exposes REST endpoints for searching, checking system status, and viewing statistics.

Start the API using:

```bash
python main.py serve --port 8000
```

Your server will be available at:

- API root: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

Example query:

```bash
curl -X POST "http://localhost:8000/search"   -H "Content-Type: application/json"   -d '{"query": "quantum physics", "top_k": 5}'
```

---

## Folder Structure

```
project_root/
├── src/
│   ├── preprocess.py          # Loads & cleans text
│   ├── cache_manager.py       # SQLite caching logic
│   ├── embedder.py            # Embedding generator
│   ├── search_engine.py       # Vector search (FAISS)
│   ├── api.py                 # FastAPI endpoints
│   └── utils.py               # Config & utilities
├── data/                      # Your .txt documents
├── cache/                     # Auto‑generated cache + FAISS index
├── main.py                    # CLI entry point
├── streamlit_ui.py           # UI application
├── requirements.txt
└── README.md
```

This structure keeps the core logic inside `src/`, while all generated files remain inside `cache/`.

---

## Design Choices

### Embeddings
The model `all-MiniLM-L6-v2` was chosen because it offers a good balance of accuracy and speed, even on CPUs.

### Search Index
FAISS IndexFlatIP was selected to keep search exact and very fast.  
It requires no training and handles millions of vectors efficiently.

### Cache Backend
SQLite was chosen for its simplicity, zero configuration, and reliable BLOB storage.

### API Framework
FastAPI provides type‑checked request models, clean async support, and automatically generated documentation.

### Interface
Streamlit was used for the UI because it allows rapid development and produces clean, modern layouts without writing frontend code manually.

---

## Running the UI

You can launch the full UI with:

```bash
streamlit run streamlit_ui.py
```

This opens a local search dashboard where you can run semantic queries, inspect search explanations, and view analytics.

---

## Installation

```bash
git clone https://github.com/yourusername/luminary.git
cd luminary
python -m venv venv
venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

Add your `.txt` documents to the `data/` folder, then run indexing or launch the UI.

---

## Final Notes

The project is intentionally kept simple. Every component is written to be readable and modifiable — suitable for students, teams, or production setups that want an understandable and extendable semantic search pipeline.

Feel free to customize the preprocessing, experiment with larger embeddings, or replace the frontend with another interface.

# Luminary-semantic-search-for-modern-knowledge-work.
