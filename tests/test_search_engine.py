"""Tests for search engine module."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.preprocess import Document
from src.search_engine import (
    FAISSIndex,
    NumpyIndex,
    QueryExpander,
    SearchEngine,
    SearchResult,
)


class TestNumpyIndex:
    """Tests for NumPy-based vector index."""
    
    def test_add_and_search(self):
        index = NumpyIndex(dimension=4)
        
        embeddings = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float32)
        
        index.add(embeddings, ["doc1", "doc2", "doc3"])
        
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        doc_ids, scores = index.search(query, top_k=2)
        
        assert len(doc_ids) == 2
        assert doc_ids[0] == "doc1"
        assert scores[0] > scores[1]
    
    def test_empty_search(self):
        index = NumpyIndex(dimension=4)
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        
        doc_ids, scores = index.search(query, top_k=5)
        
        assert len(doc_ids) == 0
        assert len(scores) == 0
    
    def test_size_property(self):
        index = NumpyIndex(dimension=4)
        assert index.size == 0
        
        embeddings = np.random.rand(5, 4).astype(np.float32)
        index.add(embeddings, [f"doc{i}" for i in range(5)])
        
        assert index.size == 5
    
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index.npz"
            
            index1 = NumpyIndex(dimension=4)
            embeddings = np.random.rand(3, 4).astype(np.float32)
            index1.add(embeddings, ["a", "b", "c"])
            index1.save(path)
            
            index2 = NumpyIndex(dimension=4)
            index2.load(path)
            
            assert index2.size == 3
            assert set(index2.doc_ids) == {"a", "b", "c"}


class TestFAISSIndex:
    """Tests for FAISS-based vector index."""
    
    @pytest.fixture(autouse=True)
    def check_faiss(self):
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")
    
    def test_add_and_search(self):
        index = FAISSIndex(dimension=4)
        
        embeddings = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float32)
        
        index.add(embeddings, ["doc1", "doc2", "doc3"])
        
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        doc_ids, scores = index.search(query, top_k=2)
        
        assert len(doc_ids) == 2
        assert doc_ids[0] == "doc1"
    
    def test_size_property(self):
        index = FAISSIndex(dimension=4)
        assert index.size == 0
        
        embeddings = np.random.rand(5, 4).astype(np.float32)
        index.add(embeddings, [f"doc{i}" for i in range(5)])
        
        assert index.size == 5


class TestSearchEngine:
    """Tests for high-level SearchEngine class."""
    
    def setup_method(self):
        self.docs = [
            Document(doc_id="physics", content="quantum physics and mechanics"),
            Document(doc_id="biology", content="biology cells and organisms"),
            Document(doc_id="chemistry", content="chemistry atoms and molecules"),
        ]
        
        self.embeddings = {
            "physics": np.array([1, 0, 0, 0], dtype=np.float32),
            "biology": np.array([0, 1, 0, 0], dtype=np.float32),
            "chemistry": np.array([0, 0, 1, 0], dtype=np.float32),
        }
    
    def test_index_documents(self):
        engine = SearchEngine(dimension=4, use_faiss=False)
        engine.index_documents(self.docs, self.embeddings)
        
        assert engine.document_count == 3
        assert "physics" in engine.documents
    
    def test_search_basic(self):
        engine = SearchEngine(dimension=4, use_faiss=False)
        engine.index_documents(self.docs, self.embeddings)
        
        query_embedding = np.array([1, 0, 0, 0], dtype=np.float32)
        response = engine.search(
            query="quantum physics",
            query_embedding=query_embedding,
            top_k=2
        )
        
        assert len(response.results) == 2
        assert response.results[0].doc_id == "physics"
        assert response.total_docs == 3
        assert response.search_time_ms >= 0
    
    def test_search_with_explanation(self):
        engine = SearchEngine(dimension=4, use_faiss=False)
        engine.index_documents(self.docs, self.embeddings)
        
        query_embedding = np.array([1, 0, 0, 0], dtype=np.float32)
        response = engine.search(
            query="physics mechanics",
            query_embedding=query_embedding,
            top_k=1,
            include_explanation=True
        )
        
        result = response.results[0]
        assert "keywords" in result.explanation
        assert "overlap_ratio" in result.explanation
        assert "doc_length" in result.explanation
    
    def test_search_length_normalize(self):
        engine = SearchEngine(dimension=4, use_faiss=False)
        engine.index_documents(self.docs, self.embeddings)
        
        query_embedding = np.array([1, 0, 0, 0], dtype=np.float32)
        response = engine.search(
            query="physics",
            query_embedding=query_embedding,
            top_k=1,
            length_normalize=True
        )
        
        result = response.results[0]
        assert "length_normalized_score" in result.explanation


class TestQueryExpander:
    """Tests for WordNet query expansion."""
    
    def test_expand_basic(self):
        expander = QueryExpander()
        
        if not expander._nltk_ready:
            pytest.skip("NLTK WordNet not available")
        
        expanded = expander.expand("happy")
        
        assert "happy" in expanded
        assert len(expanded.split()) >= 1
    
    def test_expand_short_words_unchanged(self):
        expander = QueryExpander()
        
        if not expander._nltk_ready:
            pytest.skip("NLTK WordNet not available")
        
        expanded = expander.expand("a an the")
        assert expanded == "a an the"
    
    def test_expand_preserves_original(self):
        expander = QueryExpander()
        original = "quantum physics research"
        expanded = expander.expand(original)
        
        for word in original.split():
            assert word in expanded