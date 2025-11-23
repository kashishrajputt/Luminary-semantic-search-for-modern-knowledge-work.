"""
Multi-Document Embedding Search Engine

A semantic search engine with intelligent caching using:
- sentence-transformers for embeddings
- FAISS for vector search
- SQLite for persistent caching
- FastAPI for REST API
"""

__version__ = "1.0.0"
__author__ = "Embedding Search Team"

from .cache_manager import CacheManager
from .embedder import EmbeddingGenerator
from .preprocess import Document, DocumentLoader, TextPreprocessor
from .search_engine import SearchEngine, SearchResult

__all__ = [
    "CacheManager",
    "EmbeddingGenerator", 
    "Document",
    "DocumentLoader",
    "TextPreprocessor",
    "SearchEngine",
    "SearchResult",
]