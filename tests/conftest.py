"""Pytest configuration and fixtures."""
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with sample documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        
        (data_dir / "physics.txt").write_text(
            "Quantum physics is the study of matter and energy at the molecular level. "
            "It explores concepts like wave-particle duality and quantum entanglement."
        )
        
        (data_dir / "biology.txt").write_text(
            "Biology is the scientific study of life and living organisms. "
            "It covers topics from cellular biology to ecology and evolution."
        )
        
        (data_dir / "chemistry.txt").write_text(
            "Chemistry examines the composition and properties of matter. "
            "It studies atoms, molecules, and their interactions."
        )
        
        yield data_dir


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_documents():
    """Sample Document objects for testing."""
    from src.preprocess import Document
    
    return [
        Document(
            doc_id="doc1",
            content="quantum physics mechanics",
            filename="doc1.txt"
        ),
        Document(
            doc_id="doc2",
            content="biology cells organisms",
            filename="doc2.txt"
        ),
        Document(
            doc_id="doc3",
            content="chemistry atoms molecules",
            filename="doc3.txt"
        ),
    ]