"""Tests for cache manager module."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.cache_manager import CacheManager, CachedEmbedding


class TestCacheManager:
    """Tests for CacheManager class."""
    
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        self.cache = CacheManager(db_path=self.db_path)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_set_and_get(self):
        embedding = np.random.rand(384).astype(np.float32)
        
        self.cache.set(
            doc_id="doc1",
            embedding=embedding,
            content_hash="hash123",
            metadata={"filename": "test.txt"}
        )
        
        cached = self.cache.get("doc1")
        
        assert cached is not None
        assert cached.doc_id == "doc1"
        assert cached.content_hash == "hash123"
        assert np.allclose(cached.embedding, embedding)
        assert cached.metadata["filename"] == "test.txt"
    
    def test_get_nonexistent(self):
        cached = self.cache.get("nonexistent")
        assert cached is None
    
    def test_is_valid_matching_hash(self):
        embedding = np.random.rand(384).astype(np.float32)
        self.cache.set("doc1", embedding, "hash123")
        
        assert self.cache.is_valid("doc1", "hash123") is True
    
    def test_is_valid_different_hash(self):
        embedding = np.random.rand(384).astype(np.float32)
        self.cache.set("doc1", embedding, "hash123")
        
        assert self.cache.is_valid("doc1", "hash456") is False
    
    def test_is_valid_missing_doc(self):
        assert self.cache.is_valid("missing", "hash123") is False
    
    def test_delete(self):
        embedding = np.random.rand(384).astype(np.float32)
        self.cache.set("doc1", embedding, "hash123")
        
        result = self.cache.delete("doc1")
        assert result is True
        assert self.cache.get("doc1") is None
    
    def test_delete_nonexistent(self):
        result = self.cache.delete("nonexistent")
        assert result is False
    
    def test_clear(self):
        embedding = np.random.rand(384).astype(np.float32)
        self.cache.set("doc1", embedding, "hash1")
        self.cache.set("doc2", embedding, "hash2")
        
        count = self.cache.clear()
        
        assert count == 2
        assert self.cache.get("doc1") is None
        assert self.cache.get("doc2") is None
    
    def test_get_all(self):
        embedding = np.random.rand(384).astype(np.float32)
        self.cache.set("doc1", embedding, "hash1")
        self.cache.set("doc2", embedding, "hash2")
        
        all_cached = self.cache.get_all()
        
        assert len(all_cached) == 2
        doc_ids = {c.doc_id for c in all_cached}
        assert doc_ids == {"doc1", "doc2"}
    
    def test_bulk_set(self):
        entries = [
            ("doc1", np.random.rand(384).astype(np.float32), "hash1", None),
            ("doc2", np.random.rand(384).astype(np.float32), "hash2", None),
            ("doc3", np.random.rand(384).astype(np.float32), "hash3", None),
        ]
        
        count = self.cache.bulk_set(entries)
        
        assert count == 3
        assert self.cache.get("doc1") is not None
        assert self.cache.get("doc2") is not None
        assert self.cache.get("doc3") is not None
    
    def test_get_stats(self):
        embedding = np.random.rand(384).astype(np.float32)
        self.cache.set("doc1", embedding, "hash1")
        
        stats = self.cache.get_stats()
        
        assert stats["total_entries"] == 1
        assert stats["db_size_bytes"] > 0
        assert "db_path" in stats
    
    def test_update_existing(self):
        emb1 = np.ones(384, dtype=np.float32)
        emb2 = np.zeros(384, dtype=np.float32)
        
        self.cache.set("doc1", emb1, "hash1")
        self.cache.set("doc1", emb2, "hash2")
        
        cached = self.cache.get("doc1")
        
        assert cached.content_hash == "hash2"
        assert np.allclose(cached.embedding, emb2)
    
    def test_get_by_hash(self):
        embedding = np.random.rand(384).astype(np.float32)
        self.cache.set("doc1", embedding, "unique_hash")
        
        cached = self.cache.get_by_hash("unique_hash")
        
        assert cached is not None
        assert cached.doc_id == "doc1"