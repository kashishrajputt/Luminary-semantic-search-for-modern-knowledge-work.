"""
Document preprocessing module.
Handles loading, cleaning, and metadata extraction for text documents.
"""
import html
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .utils import Config, compute_hash, get_logger, timer

logger = get_logger(__name__)


@dataclass
class Document:
    """Represents a preprocessed document with metadata."""
    doc_id: str
    content: str
    original_content: str = ""
    filename: str = ""
    doc_length: int = 0
    content_hash: str = ""
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if not self.doc_length:
            self.doc_length = len(self.content)
        if not self.content_hash:
            self.content_hash = compute_hash(self.content)


class TextPreprocessor:
    """
    Text preprocessing pipeline for document cleaning.
    
    Pipeline steps:
    1. Lowercase conversion
    2. HTML tag removal
    3. Extra whitespace normalization
    """
    
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    
    def __init__(self, remove_urls: bool = True) -> None:
        self.remove_urls = remove_urls
    
    def clean(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        text = self._lowercase(text)
        text = self._remove_html(text)
        if self.remove_urls:
            text = self._remove_urls(text)
        text = self._normalize_whitespace(text)
        return text.strip()
    
    def _lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and unescape HTML entities."""
        text = self.HTML_TAG_PATTERN.sub(" ", text)
        text = html.unescape(text)
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.URL_PATTERN.sub(" ", text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize multiple whitespace to single space."""
        return self.WHITESPACE_PATTERN.sub(" ", text)


class DocumentLoader:
    """
    Loads and preprocesses documents from a directory.
    
    Supports batch loading with optional multiprocessing.
    """
    
    SUPPORTED_EXTENSIONS = {".txt"}
    
    def __init__(
        self,
        data_dir: str | Path = Config.DATA_DIR,
        preprocessor: TextPreprocessor | None = None,
        use_multiprocessing: bool = False,
        max_workers: int = Config.MAX_WORKERS
    ) -> None:
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor or TextPreprocessor()
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
    
    def _get_files(self) -> list[Path]:
        """Get all supported files in data directory."""
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.data_dir.glob(f"*{ext}"))
        return sorted(files)
    
    def _load_single(self, filepath: Path) -> Document | None:
        """Load and preprocess a single document."""
        try:
            original = filepath.read_text(encoding="utf-8", errors="ignore")
            cleaned = self.preprocessor.clean(original)
            
            if not cleaned:
                logger.warning(f"Empty content after preprocessing: {filepath.name}")
                return None
            
            return Document(
                doc_id=filepath.stem,
                content=cleaned,
                original_content=original,
                filename=filepath.name,
                metadata={"source_path": str(filepath)}
            )
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None
    
    @timer
    def load_all(self) -> list[Document]:
        """
        Load all documents from data directory.
        
        Returns:
            List of preprocessed Document objects
        """
        files = self._get_files()
        
        if not files:
            logger.warning(f"No documents found in {self.data_dir}")
            return []
        
        logger.info(f"Loading {len(files)} documents from {self.data_dir}")
        
        if self.use_multiprocessing and len(files) > 10:
            return self._load_parallel(files)
        return self._load_sequential(files)
    
    def _load_sequential(self, files: list[Path]) -> list[Document]:
        """Load documents sequentially."""
        docs = []
        for f in files:
            doc = self._load_single(f)
            if doc:
                docs.append(doc)
        logger.info(f"Loaded {len(docs)} documents successfully")
        return docs
    
    def _load_parallel(self, files: list[Path]) -> list[Document]:
        """Load documents in parallel using multiprocessing."""
        docs = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._load_single, f): f for f in files}
            for future in as_completed(futures):
                doc = future.result()
                if doc:
                    docs.append(doc)
        logger.info(f"Loaded {len(docs)} documents in parallel")
        return docs
    
    def load_iterator(self) -> Iterator[Document]:
        """Yield documents one at a time for memory efficiency."""
        for filepath in self._get_files():
            doc = self._load_single(filepath)
            if doc:
                yield doc


def preprocess_documents(
    data_dir: str | Path = Config.DATA_DIR,
    use_multiprocessing: bool = False
) -> list[Document]:
    """
    Convenience function to load and preprocess all documents.
    
    Args:
        data_dir: Directory containing .txt files
        use_multiprocessing: Enable parallel loading
        
    Returns:
        List of preprocessed documents
    """
    loader = DocumentLoader(
        data_dir=data_dir,
        use_multiprocessing=use_multiprocessing
    )
    return loader.load_all()