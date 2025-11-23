"""
Embedding generation module using sentence-transformers.
Includes caching integration and batch processing support.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .cache_manager import CacheManager
from .preprocess import Document
from .utils import Config, get_logger, timer

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers with cache integration.
    
    Features:
    - Automatic caching with hash validation
    - Batch processing for efficiency
    - GPU support when available
    - Normalized embeddings for cosine similarity
    
    Example:
        >>> embedder = EmbeddingGenerator()
        >>> embedding = embedder.embed_text("Hello world")
        >>> print(embedding.shape)  # (384,)
    """
    
    def __init__(
        self,
        model_name: str = Config.EMBEDDING_MODEL,
        cache_manager: CacheManager | None = None,
        use_cache: bool = True,
        batch_size: int = Config.BATCH_SIZE,
        normalize: bool = True
    ) -> None:
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model identifier
            cache_manager: Optional cache manager instance
            use_cache: Whether to use caching
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.use_cache = use_cache
        
        # Load model
        logger.info(f"Loading embedding model: {model_name}")
        device = "cuda" if Config.USE_GPU else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize cache
        self.cache = cache_manager if use_cache else None
        if use_cache and not cache_manager:
            self.cache = CacheManager()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    @timer
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return np.array(embedding, dtype=np.float32)
    
    @timer
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: Sequence of input texts
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)
        
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100
        )
        return np.array(embeddings, dtype=np.float32)
    
    def embed_document(self, doc: Document, force: bool = False) -> np.ndarray:
        """
        Generate embedding for a document with caching.
        
        Args:
            doc: Document to embed
            force: Force regeneration even if cached
            
        Returns:
            Document embedding
        """
        # Check cache first
        if self.cache and not force:
            if self.cache.is_valid(doc.doc_id, doc.content_hash):
                cached = self.cache.get(doc.doc_id)
                if cached:
                    logger.debug(f"Cache hit for {doc.doc_id}")
                    return cached.embedding
        
        # Generate new embedding
        logger.debug(f"Generating embedding for {doc.doc_id}")
        embedding = self.embed_text(doc.content)
        
        # Store in cache
        if self.cache:
            self.cache.set(
                doc_id=doc.doc_id,
                embedding=embedding,
                content_hash=doc.content_hash,
                metadata={
                    "filename": doc.filename,
                    "doc_length": doc.doc_length
                }
            )
        
        return embedding
    
    @timer
    def embed_documents(
        self,
        documents: list[Document],
        force: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Generate embeddings for multiple documents with smart caching.
        
        This method implements the caching logic:
        1. Check each document against cache
        2. Batch embed only documents that need new embeddings
        3. Store new embeddings in cache
        
        Args:
            documents: List of documents to embed
            force: Force regeneration for all documents
            
        Returns:
            Dictionary mapping doc_id to embedding
        """
        results: dict[str, np.ndarray] = {}
        to_embed: list[Document] = []
        
        # Phase 1: Check cache for existing embeddings
        if self.cache and not force:
            for doc in documents:
                if self.cache.is_valid(doc.doc_id, doc.content_hash):
                    cached = self.cache.get(doc.doc_id)
                    if cached:
                        results[doc.doc_id] = cached.embedding
                        continue
                to_embed.append(doc)
            
            logger.info(
                f"Cache: {len(results)} hits, {len(to_embed)} misses"
            )
        else:
            to_embed = documents
        
        # Phase 2: Batch embed documents that need new embeddings
        if to_embed:
            texts = [d.content for d in to_embed]
            embeddings = self.embed_texts(texts)
            
            # Prepare cache entries
            cache_entries = []
            for doc, emb in zip(to_embed, embeddings):
                results[doc.doc_id] = emb
                if self.cache:
                    cache_entries.append((
                        doc.doc_id,
                        emb,
                        doc.content_hash,
                        {"filename": doc.filename, "doc_length": doc.doc_length}
                    ))
            
            # Phase 3: Bulk update cache
            if self.cache and cache_entries:
                self.cache.bulk_set(cache_entries)
        
        return results
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query.
        
        Queries are not cached as they are typically unique.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding
        """
        return self.embed_text(query)


class BatchEmbedder:
    """
    Multiprocessing-enabled batch embedder for large document collections.
    
    Note: Due to model loading overhead, this is most efficient for
    very large collections (1000+ documents).
    
    Example:
        >>> embedder = BatchEmbedder()
        >>> embeddings = embedder.embed_parallel(texts)
    """
    
    def __init__(
        self,
        model_name: str = Config.EMBEDDING_MODEL,
        max_workers: int = Config.MAX_WORKERS,
        batch_size: int = Config.BATCH_SIZE
    ) -> None:
        """
        Initialize batch embedder.
        
        Args:
            model_name: HuggingFace model identifier
            max_workers: Maximum parallel workers
            batch_size: Batch size per worker
        """
        self.model_name = model_name
        self.max_workers = max_workers
        self.batch_size = batch_size
    
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts (runs in worker thread).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Batch embeddings
        """
        model = SentenceTransformer(self.model_name)
        return model.encode(texts, normalize_embeddings=True)
    
    def embed_parallel(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts using parallel processing.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Stacked embeddings array
        """
        # Fall back to sequential for small batches
        if len(texts) < self.batch_size * 2:
            model = SentenceTransformer(self.model_name)
            return model.encode(texts, normalize_embeddings=True)
        
        # Split into batches
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        # Process in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._embed_batch, b) for b in batches]
            for future in futures:
                results.append(future.result())
        
        return np.vstack(results)