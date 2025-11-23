"""
Vector search engine using FAISS with NumPy fallback.
Provides semantic search with ranking explanations.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .preprocess import Document
from .utils import (
    Config,
    calculate_overlap,
    extract_keywords,
    get_logger,
    timer,
    truncate_text,
)

logger = get_logger(__name__)

# Try to import FAISS, fall back to NumPy if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using NumPy fallback")


@dataclass
class SearchResult:
    """
    Represents a single search result with explanation.
    
    Attributes:
        doc_id: Document identifier
        score: Similarity score
        preview: Text preview snippet
        explanation: Ranking explanation details
        metadata: Additional metadata
    """
    doc_id: str
    score: float
    preview: str
    explanation: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResponse:
    """
    Complete search response.
    
    Attributes:
        results: List of search results
        query: Original query text
        total_docs: Total documents in index
        search_time_ms: Search execution time
    """
    results: list[SearchResult]
    query: str
    total_docs: int
    search_time_ms: float


class VectorIndex:
    """
    Abstract vector index interface.
    
    Implementations: FAISSIndex, NumpyIndex
    """
    
    def add(self, embeddings: np.ndarray, doc_ids: list[str]) -> None:
        """Add embeddings to index."""
        raise NotImplementedError
    
    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        raise NotImplementedError
    
    def save(self, path: Path) -> None:
        """Save index to disk."""
        raise NotImplementedError
    
    def load(self, path: Path) -> None:
        """Load index from disk."""
        raise NotImplementedError


class FAISSIndex(VectorIndex):
    """
    FAISS-based vector index using IndexFlatIP (inner product).
    
    Inner product on normalized vectors equals cosine similarity.
    """
    
    def __init__(self, dimension: int = Config.EMBEDDING_DIM) -> None:
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.doc_ids: list[str] = []
    
    def add(self, embeddings: np.ndarray, doc_ids: list[str]) -> None:
        """
        Add embeddings to index.
        
        Args:
            embeddings: Array of shape (n, dimension)
            doc_ids: List of document identifiers
        """
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError("Embeddings count must match doc_ids count")
        
        # Ensure float32 and normalize
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)
        logger.info(f"Added {len(doc_ids)} vectors to FAISS index")
    
    def search(
        self, 
        query: np.ndarray, 
        top_k: int = 5
    ) -> tuple[list[str], np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query: Query embedding
            top_k: Number of results
            
        Returns:
            Tuple of (doc_ids, scores)
        """
        # Prepare query
        query = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search
        k = min(top_k, len(self.doc_ids))
        scores, indices = self.index.search(query, k)
        
        # Map indices to doc_ids
        result_ids = [self.doc_ids[i] for i in indices[0] if i >= 0]
        return result_ids, scores[0][:len(result_ids)]
    
    def save(self, path: Path) -> None:
        """Save index to disk."""
        faiss.write_index(self.index, str(path))
        # Save doc_ids separately
        np.save(str(path) + ".ids.npy", np.array(self.doc_ids, dtype=object))
        logger.info(f"Saved FAISS index to {path}")
    
    def load(self, path: Path) -> None:
        """Load index from disk."""
        self.index = faiss.read_index(str(path))
        self.doc_ids = list(np.load(str(path) + ".ids.npy", allow_pickle=True))
        logger.info(f"Loaded FAISS index with {len(self.doc_ids)} vectors")
    
    @property
    def size(self) -> int:
        """Number of vectors in index."""
        return self.index.ntotal


class NumpyIndex(VectorIndex):
    """
    NumPy-based fallback using cosine similarity.
    
    Use when FAISS is not available.
    """
    
    def __init__(self, dimension: int = Config.EMBEDDING_DIM) -> None:
        """
        Initialize NumPy index.
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.embeddings: np.ndarray | None = None
        self.doc_ids: list[str] = []
    
    def add(self, embeddings: np.ndarray, doc_ids: list[str]) -> None:
        """Add embeddings to index."""
        # Normalize embeddings
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.doc_ids.extend(doc_ids)
        logger.info(f"Added {len(doc_ids)} vectors to NumPy index")
    
    def search(
        self, 
        query: np.ndarray, 
        top_k: int = 5
    ) -> tuple[list[str], np.ndarray]:
        """Search using cosine similarity."""
        if self.embeddings is None or len(self.doc_ids) == 0:
            return [], np.array([])
        
        # Normalize query
        query = query.astype(np.float32).flatten()
        query = query / (np.linalg.norm(query) + 1e-8)
        
        # Compute similarities (dot product of normalized vectors = cosine)
        scores = self.embeddings @ query
        
        # Get top-k
        k = min(top_k, len(self.doc_ids))
        top_indices = np.argsort(scores)[::-1][:k]
        
        result_ids = [self.doc_ids[i] for i in top_indices]
        return result_ids, scores[top_indices]
    
    def save(self, path: Path) -> None:
        """Save index to disk."""
        np.savez(
            path,
            embeddings=self.embeddings,
            doc_ids=np.array(self.doc_ids, dtype=object)
        )
        logger.info(f"Saved NumPy index to {path}")
    
    def load(self, path: Path) -> None:
        """Load index from disk."""
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.doc_ids = list(data["doc_ids"])
        logger.info(f"Loaded NumPy index with {len(self.doc_ids)} vectors")
    
    @property
    def size(self) -> int:
        """Number of vectors in index."""
        return len(self.doc_ids)


class SearchEngine:
    """
    High-level search engine with ranking explanations.
    
    Features:
    - FAISS or NumPy backend
    - Keyword overlap analysis
    - Document length normalization
    - Persistent index support
    
    Example:
        >>> engine = SearchEngine()
        >>> engine.index_documents(docs, embeddings)
        >>> response = engine.search("quantum physics", query_embedding)
        >>> for result in response.results:
        ...     print(f"{result.doc_id}: {result.score}")
    """
    
    def __init__(
        self,
        dimension: int = Config.EMBEDDING_DIM,
        use_faiss: bool = True,
        index_path: Path | None = None
    ) -> None:
        """
        Initialize search engine.
        
        Args:
            dimension: Embedding dimension
            use_faiss: Whether to use FAISS (falls back to NumPy if unavailable)
            index_path: Path for persistent index
        """
        self.dimension = dimension
        self.documents: dict[str, Document] = {}
        
        # Choose index backend
        if use_faiss and FAISS_AVAILABLE:
            self.index: VectorIndex = FAISSIndex(dimension)
        else:
            self.index = NumpyIndex(dimension)
        
        self.index_path = index_path or Config.FAISS_INDEX_PATH
        
        # Try to load existing index
        if self.index_path.exists():
            try:
                self.index.load(self.index_path)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
    
    def index_documents(
        self,
        documents: list[Document],
        embeddings: dict[str, np.ndarray]
    ) -> None:
        """
        Index documents with their embeddings.
        
        Args:
            documents: List of documents
            embeddings: Dict mapping doc_id to embedding
        """
        # Store documents for retrieval
        for doc in documents:
            self.documents[doc.doc_id] = doc
        
        # Build index
        ordered_ids = list(embeddings.keys())
        ordered_embeds = np.array([embeddings[did] for did in ordered_ids])
        
        # Reset index
        self.index = (
            FAISSIndex(self.dimension) 
            if FAISS_AVAILABLE 
            else NumpyIndex(self.dimension)
        )
        self.index.add(ordered_embeds, ordered_ids)
    
    def save_index(self) -> None:
        """Persist index to disk."""
        self.index.save(self.index_path)
    
    @timer
    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        include_explanation: bool = True,
        length_normalize: bool = False
    ) -> SearchResponse:
        """
        Search for documents similar to query.
        
        Args:
            query: Original query text
            query_embedding: Embedded query vector
            top_k: Number of results to return
            include_explanation: Include ranking explanation
            length_normalize: Apply document length normalization
            
        Returns:
            SearchResponse with results and metadata
        """
        import time
        start = time.perf_counter()
        
        # Perform vector search
        doc_ids, scores = self.index.search(query_embedding, top_k)
        
        # Extract query keywords for explanation
        query_keywords = extract_keywords(query)
        results = []
        
        for doc_id, score in zip(doc_ids, scores):
            doc = self.documents.get(doc_id)
            if not doc:
                continue
            
            # Apply length normalization if requested
            final_score = float(score)
            if length_normalize and doc.doc_length > 0:
                # Penalize very long documents slightly
                length_factor = 1.0 / (1.0 + np.log1p(doc.doc_length / 1000))
                final_score *= length_factor
            
            # Build explanation
            explanation = {}
            if include_explanation:
                doc_keywords = extract_keywords(doc.content[:500])
                overlap = query_keywords & doc_keywords
                overlap_ratio = calculate_overlap(query_keywords, doc_keywords)
                
                explanation = {
                    "keywords": list(overlap)[:10],
                    "overlap_ratio": round(overlap_ratio, 4),
                    "doc_length": doc.doc_length,
                    "raw_score": round(float(score), 4),
                }
                if length_normalize:
                    explanation["length_normalized_score"] = round(final_score, 4)
            
            results.append(SearchResult(
                doc_id=doc_id,
                score=round(final_score, 4),
                preview=truncate_text(doc.content, 200),
                explanation=explanation,
                metadata={"filename": doc.filename}
            ))
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return SearchResponse(
            results=results,
            query=query,
            total_docs=self.index.size,
            search_time_ms=round(elapsed, 2)
        )
    
    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return len(self.documents)


class QueryExpander:
    """
    Query expansion using WordNet synonyms.
    
    Improves recall by including related terms in the query.
    
    Example:
        >>> expander = QueryExpander()
        >>> expanded = expander.expand("happy")
        >>> print(expanded)  # "happy glad joyful"
    """
    
    def __init__(self) -> None:
        """Initialize query expander."""
        self._nltk_ready = False
        self._init_nltk()
    
    def _init_nltk(self) -> None:
        """Initialize NLTK WordNet."""
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet
            self.wordnet = wordnet
            self._nltk_ready = True
        except Exception as e:
            logger.warning(f"WordNet initialization failed: {e}")
    
    def expand(self, query: str, max_synonyms: int = 2) -> str:
        """
        Expand query with synonyms.
        
        Args:
            query: Original query
            max_synonyms: Max synonyms per word
            
        Returns:
            Expanded query string
        """
        if not self._nltk_ready:
            return query
        
        words = query.lower().split()
        expanded = list(words)
        
        for word in words:
            # Skip short words
            if len(word) < 4:
                continue
            
            # Get synsets
            synsets = self.wordnet.synsets(word)
            synonyms = set()
            
            for syn in synsets[:3]:
                for lemma in syn.lemmas()[:max_synonyms]:
                    name = lemma.name().replace("_", " ").lower()
                    if name != word and name not in expanded:
                        synonyms.add(name)
            
            expanded.extend(list(synonyms)[:max_synonyms])
        
        return " ".join(expanded)