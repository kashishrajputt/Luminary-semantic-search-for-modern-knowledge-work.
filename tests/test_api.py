"""Tests for FastAPI endpoints."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app, state, initialize_engine
from src.preprocess import Document
from src.search_engine import SearchEngine, SearchResponse, SearchResult


@pytest.fixture
def client():
    """Create test client with mocked state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        cache_dir = Path(tmpdir) / "cache"
        data_dir.mkdir()
        cache_dir.mkdir()
        
        (data_dir / "test1.txt").write_text("Quantum physics introduction")
        (data_dir / "test2.txt").write_text("Biology and life sciences")
        
        with patch("src.api.Config") as mock_config:
            mock_config.DATA_DIR = data_dir
            mock_config.CACHE_DIR = cache_dir
            mock_config.CACHE_DB_PATH = cache_dir / "test.db"
            mock_config.FAISS_INDEX_PATH = cache_dir / "test.index"
            mock_config.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            mock_config.EMBEDDING_DIM = 384
            mock_config.USE_GPU = False
            mock_config.BATCH_SIZE = 32
            mock_config.init_dirs = lambda: None
            
            state.initialized = False
            
            with TestClient(app) as c:
                yield c


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "index_loaded" in data


class TestSearchEndpoint:
    """Tests for /search endpoint."""
    
    def test_search_basic(self, client):
        response = client.post(
            "/search",
            json={"query": "physics", "top_k": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "query" in data
        assert "total_docs" in data
        assert "search_time_ms" in data
    
    def test_search_with_explanation(self, client):
        response = client.post(
            "/search",
            json={"query": "quantum mechanics", "top_k": 1}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        if data["results"]:
            result = data["results"][0]
            assert "explanation" in result
            assert "keywords" in result["explanation"]
            assert "overlap_ratio" in result["explanation"]
    
    def test_search_empty_query(self, client):
        response = client.post(
            "/search",
            json={"query": "", "top_k": 5}
        )
        
        assert response.status_code == 422
    
    def test_search_invalid_top_k(self, client):
        response = client.post(
            "/search",
            json={"query": "test", "top_k": 0}
        )
        
        assert response.status_code == 422
    
    def test_search_with_options(self, client):
        response = client.post(
            "/search",
            json={
                "query": "science",
                "top_k": 3,
                "expand_query": True,
                "length_normalize": True
            }
        )
        
        assert response.status_code == 200


class TestStatsEndpoint:
    """Tests for /stats endpoint."""
    
    def test_get_stats(self, client):
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "indexed_documents" in data
        assert "cached_embeddings" in data
        assert "cache_db_size_bytes" in data
        assert "index_type" in data


class TestDocumentsEndpoint:
    """Tests for /documents endpoint."""
    
    def test_list_documents(self, client):
        response = client.get("/documents")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)


class TestReindexEndpoint:
    """Tests for /reindex endpoint."""
    
    def test_reindex(self, client):
        response = client.post(
            "/reindex",
            json={"force": False}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "documents_indexed" in data
    
    def test_reindex_force(self, client):
        response = client.post(
            "/reindex",
            json={"force": True}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["force"] is True