"""
Shared utilities for the embedding search engine.
"""
import hashlib
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


T = TypeVar("T")


def compute_hash(text: str) -> str:
    """
    Compute SHA256 hash of text content.
    
    Args:
        text: Input text to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timer(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger = get_logger(func.__module__)
        logger.debug(f"{func.__name__} executed in {elapsed:.2f}ms")
        return result
    return wrapper


def get_env(key: str, default: str = "") -> str:
    """
    Get environment variable with default.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Environment variable value or default
    """
    return os.environ.get(key, default)


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize score to [0, 1] range.
    
    Args:
        score: Raw score
        min_val: Minimum expected value
        max_val: Maximum expected value
        
    Returns:
        Normalized score between 0 and 1
    """
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to max length with suffix.
    
    Args:
        text: Input text
        max_length: Maximum length including suffix
        suffix: Suffix to append when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)].rsplit(" ", 1)[0] + suffix


def extract_keywords(text: str, min_length: int = 3) -> set[str]:
    """
    Extract unique keywords from text.
    
    Args:
        text: Input text
        min_length: Minimum word length to include
        
    Returns:
        Set of unique keywords
    """
    words = text.lower().split()
    return {w for w in words if len(w) >= min_length and w.isalpha()}


def calculate_overlap(set1: set[str], set2: set[str]) -> float:
    """
    Calculate Jaccard-like overlap ratio between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Overlap ratio between 0 and 1
    """
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


class Config:
    """Application configuration with environment variable support."""
    
    # Model settings
    EMBEDDING_MODEL: str = get_env(
        "EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIM: int = 384
    
    # Paths
    CACHE_DIR: Path = Path(get_env("CACHE_DIR", "./cache"))
    DATA_DIR: Path = Path(get_env("DATA_DIR", "./data"))
    FAISS_INDEX_PATH: Path = CACHE_DIR / "faiss.index"
    CACHE_DB_PATH: Path = CACHE_DIR / "embeddings.db"
    
    # Processing settings
    USE_GPU: bool = get_env("USE_GPU", "false").lower() == "true"
    BATCH_SIZE: int = int(get_env("BATCH_SIZE", "32"))
    MAX_WORKERS: int = int(get_env("MAX_WORKERS", "4"))
    
    @classmethod
    def init_dirs(cls) -> None:
        """Initialize required directories."""
        ensure_dir(cls.CACHE_DIR)
        ensure_dir(cls.DATA_DIR)