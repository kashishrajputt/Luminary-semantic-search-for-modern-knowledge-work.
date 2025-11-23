"""
Embedding cache manager using SQLite.
Provides persistent storage with hash-based validation for embeddings.
"""
import json
import pickle
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from .utils import Config, ensure_dir, get_logger

logger = get_logger(__name__)


@dataclass
class CachedEmbedding:
    """
    Represents a cached embedding entry.
    
    Attributes:
        doc_id: Document identifier
        embedding: Numpy array of embedding vector
        content_hash: SHA256 hash of document content
        updated_at: Timestamp of last update
        metadata: Optional metadata dictionary
    """
    doc_id: str
    embedding: np.ndarray
    content_hash: str
    updated_at: datetime
    metadata: dict | None = None


class CacheManager:
    """
    SQLite-based cache for document embeddings.
    
    Features:
    - Hash-based validation (only regenerate on content change)
    - BLOB storage for efficient embedding persistence
    - WAL mode for concurrent read access
    - Automatic schema migration
    
    Example:
        >>> cache = CacheManager()
        >>> cache.set("doc1", embedding, "abc123hash")
        >>> cached = cache.get("doc1")
        >>> if cache.is_valid("doc1", current_hash):
        ...     print("Cache hit!")
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS embeddings (
        doc_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL,
        content_hash TEXT NOT NULL,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_hash ON embeddings(content_hash);
    """
    
    def __init__(self, db_path: str | Path = Config.CACHE_DB_PATH) -> None:
        """
        Initialize cache manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
    
    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """
        Context manager for database connections.
        
        Yields:
            SQLite connection with auto-commit/rollback
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get(self, doc_id: str) -> CachedEmbedding | None:
        """
        Retrieve cached embedding by document ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            CachedEmbedding if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM embeddings WHERE doc_id = ?",
                (doc_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return CachedEmbedding(
                doc_id=row["doc_id"],
                embedding=pickle.loads(row["embedding"]),
                content_hash=row["content_hash"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else None
            )
    
    def get_by_hash(self, content_hash: str) -> CachedEmbedding | None:
        """
        Retrieve cached embedding by content hash.
        
        Args:
            content_hash: SHA256 hash of document content
            
        Returns:
            CachedEmbedding if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM embeddings WHERE content_hash = ?",
                (content_hash,)
            ).fetchone()
            
            if not row:
                return None
            
            return CachedEmbedding(
                doc_id=row["doc_id"],
                embedding=pickle.loads(row["embedding"]),
                content_hash=row["content_hash"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else None
            )
    
    def set(
        self,
        doc_id: str,
        embedding: np.ndarray,
        content_hash: str,
        metadata: dict | None = None
    ) -> None:
        """
        Store or update embedding in cache.
        
        Args:
            doc_id: Document identifier
            embedding: Numpy array of embedding vector
            content_hash: SHA256 hash of document content
            metadata: Optional metadata dictionary
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings 
                (doc_id, embedding, content_hash, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    pickle.dumps(embedding),
                    content_hash,
                    datetime.utcnow().isoformat(),
                    json.dumps(metadata) if metadata else None
                )
            )
        logger.debug(f"Cached embedding for {doc_id}")
    
    def is_valid(self, doc_id: str, content_hash: str) -> bool:
        """
        Check if cached embedding is valid (hash matches).
        
        This is the core caching logic:
        - If hash matches: cache is valid, reuse embedding
        - If hash differs: cache is stale, regenerate embedding
        
        Args:
            doc_id: Document identifier
            content_hash: Current content hash to validate against
            
        Returns:
            True if cache is valid, False if missing or stale
        """
        cached = self.get(doc_id)
        if not cached:
            return False
        return cached.content_hash == content_hash
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete embedding from cache.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM embeddings WHERE doc_id = ?",
                (doc_id,)
            )
            return cursor.rowcount > 0
    
    def clear(self) -> int:
        """
        Clear all cached embeddings.
        
        Returns:
            Number of entries deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM embeddings")
            count = cursor.rowcount
        logger.info(f"Cleared {count} cached embeddings")
        return count
    
    def get_all(self) -> list[CachedEmbedding]:
        """
        Retrieve all cached embeddings.
        
        Returns:
            List of all cached embeddings
        """
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM embeddings").fetchall()
            return [
                CachedEmbedding(
                    doc_id=row["doc_id"],
                    embedding=pickle.loads(row["embedding"]),
                    content_hash=row["content_hash"],
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None
                )
                for row in rows
            ]
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self._get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM embeddings"
            ).fetchone()["cnt"]
            
            latest = conn.execute(
                "SELECT MAX(updated_at) as latest FROM embeddings"
            ).fetchone()["latest"]
            
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
        return {
            "total_entries": count,
            "latest_update": latest,
            "db_size_bytes": db_size,
            "db_path": str(self.db_path)
        }
    
    def bulk_set(
        self,
        entries: list[tuple[str, np.ndarray, str, dict | None]]
    ) -> int:
        """
        Bulk insert/update embeddings.
        
        More efficient than individual set() calls for large batches.
        
        Args:
            entries: List of (doc_id, embedding, hash, metadata) tuples
            
        Returns:
            Number of entries inserted
        """
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings 
                (doc_id, embedding, content_hash, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        doc_id,
                        pickle.dumps(emb),
                        h,
                        datetime.utcnow().isoformat(),
                        json.dumps(meta) if meta else None
                    )
                    for doc_id, emb, h, meta in entries
                ]
            )
        logger.info(f"Bulk cached {len(entries)} embeddings")
        return len(entries)