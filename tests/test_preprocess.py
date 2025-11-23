"""Tests for preprocessing module."""
import tempfile
from pathlib import Path

import pytest

from src.preprocess import Document, DocumentLoader, TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""
    
    def setup_method(self):
        self.preprocessor = TextPreprocessor()
    
    def test_lowercase(self):
        text = "Hello WORLD Test"
        result = self.preprocessor.clean(text)
        assert result == "hello world test"
    
    def test_remove_html_tags(self):
        text = "<p>Hello <strong>World</strong></p>"
        result = self.preprocessor.clean(text)
        assert "<" not in result
        assert ">" not in result
        assert "hello" in result
        assert "world" in result
    
    def test_html_entities(self):
        text = "Hello &amp; World &lt;test&gt;"
        result = self.preprocessor.clean(text)
        assert "&amp;" not in result
        assert "&" in result or "and" in result.lower()
    
    def test_normalize_whitespace(self):
        text = "Hello    World\n\n\tTest"
        result = self.preprocessor.clean(text)
        assert "  " not in result
        assert "\n" not in result
        assert "\t" not in result
    
    def test_remove_urls(self):
        text = "Visit https://example.com for more info"
        result = self.preprocessor.clean(text)
        assert "https://" not in result
        assert "example.com" not in result
    
    def test_empty_string(self):
        result = self.preprocessor.clean("")
        assert result == ""
    
    def test_combined_cleaning(self):
        text = "<div>Hello   WORLD</div> visit https://test.com &amp; more"
        result = self.preprocessor.clean(text)
        assert result == "hello world visit & more"


class TestDocument:
    """Tests for Document dataclass."""
    
    def test_auto_hash(self):
        doc = Document(doc_id="test", content="hello world")
        assert doc.content_hash != ""
        assert len(doc.content_hash) == 64
    
    def test_auto_length(self):
        doc = Document(doc_id="test", content="hello world")
        assert doc.doc_length == 11
    
    def test_same_content_same_hash(self):
        doc1 = Document(doc_id="test1", content="hello world")
        doc2 = Document(doc_id="test2", content="hello world")
        assert doc1.content_hash == doc2.content_hash
    
    def test_different_content_different_hash(self):
        doc1 = Document(doc_id="test1", content="hello world")
        doc2 = Document(doc_id="test2", content="hello universe")
        assert doc1.content_hash != doc2.content_hash


class TestDocumentLoader:
    """Tests for DocumentLoader class."""
    
    def test_load_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "doc1.txt").write_text("Hello World")
            (p / "doc2.txt").write_text("Test Document")
            
            loader = DocumentLoader(data_dir=p)
            docs = loader.load_all()
            
            assert len(docs) == 2
            assert all(isinstance(d, Document) for d in docs)
    
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DocumentLoader(data_dir=tmpdir)
            docs = loader.load_all()
            assert len(docs) == 0
    
    def test_skip_empty_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "empty.txt").write_text("")
            (p / "valid.txt").write_text("Content")
            
            loader = DocumentLoader(data_dir=p)
            docs = loader.load_all()
            
            assert len(docs) == 1
            assert docs[0].doc_id == "valid"
    
    def test_preprocessing_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "test.txt").write_text("<p>HELLO   WORLD</p>")
            
            loader = DocumentLoader(data_dir=p)
            docs = loader.load_all()
            
            assert docs[0].content == "hello world"
    
    def test_iterator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "doc1.txt").write_text("Doc 1")
            (p / "doc2.txt").write_text("Doc 2")
            
            loader = DocumentLoader(data_dir=p)
            docs = list(loader.load_iterator())
            
            assert len(docs) == 2