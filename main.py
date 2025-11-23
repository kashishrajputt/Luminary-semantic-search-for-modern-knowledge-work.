#!/usr/bin/env python3
"""
CLI entry point for the embedding search engine.

Usage:
    python main.py index --data-dir ./data
    python main.py serve --host 0.0.0.0 --port 8000
    python main.py search "quantum physics"
"""
import argparse
import sys
from pathlib import Path

from src.cache_manager import CacheManager
from src.embedder import EmbeddingGenerator
from src.preprocess import preprocess_documents
from src.search_engine import QueryExpander, SearchEngine
from src.utils import Config, get_logger

logger = get_logger(__name__)


def cmd_index(args: argparse.Namespace) -> int:
    """Index documents from data directory."""
    Config.init_dirs()
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    logger.info(f"Loading documents from {data_dir}")
    docs = preprocess_documents(data_dir, use_multiprocessing=args.parallel)
    
    if not docs:
        logger.warning("No documents found")
        return 0
    
    logger.info(f"Found {len(docs)} documents")
    
    cache = CacheManager()
    embedder = EmbeddingGenerator(cache_manager=cache)
    
    logger.info("Generating embeddings...")
    embeddings = embedder.embed_documents(docs, force=args.force)
    
    logger.info("Building search index...")
    engine = SearchEngine(dimension=embedder.embedding_dim)
    engine.index_documents(docs, embeddings)
    engine.save_index()
    
    cache_stats = cache.get_stats()
    logger.info(f"Indexing complete:")
    logger.info(f"  Documents indexed: {len(docs)}")
    logger.info(f"  Embeddings cached: {cache_stats['total_entries']}")
    logger.info(f"  Cache size: {cache_stats['db_size_bytes'] / 1024:.1f} KB")
    
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn")
        return 1
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Perform a search query."""
    Config.init_dirs()
    
    cache = CacheManager()
    embedder = EmbeddingGenerator(cache_manager=cache)
    engine = SearchEngine(dimension=embedder.embedding_dim)
    
    if engine.document_count == 0:
        docs = preprocess_documents(Config.DATA_DIR)
        if docs:
            embeddings = embedder.embed_documents(docs)
            engine.index_documents(docs, embeddings)
        else:
            logger.error("No documents indexed. Run 'index' command first.")
            return 1
    
    query = args.query
    
    if args.expand:
        expander = QueryExpander()
        query = expander.expand(query)
        logger.info(f"Expanded query: {query}")
    
    query_embedding = embedder.embed_query(query)
    response = engine.search(
        query=args.query,
        query_embedding=query_embedding,
        top_k=args.top_k
    )
    
    print(f"\nSearch Results for: '{args.query}'")
    print(f"Total documents: {response.total_docs}")
    print(f"Search time: {response.search_time_ms:.2f}ms")
    print("-" * 60)
    
    for i, r in enumerate(response.results, 1):
        print(f"\n{i}. {r.doc_id} (score: {r.score:.4f})")
        print(f"   Preview: {r.preview}")
        if r.explanation:
            kw = ", ".join(r.explanation.get("keywords", [])[:5])
            print(f"   Keywords: {kw}")
            print(f"   Overlap: {r.explanation.get('overlap_ratio', 0):.2%}")
    
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show cache and index statistics."""
    Config.init_dirs()
    
    cache = CacheManager()
    stats = cache.get_stats()
    
    print("\nCache Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Database size: {stats['db_size_bytes'] / 1024:.1f} KB")
    print(f"  Last update: {stats['latest_update']}")
    print(f"  DB path: {stats['db_path']}")
    
    if Config.FAISS_INDEX_PATH.exists():
        idx_size = Config.FAISS_INDEX_PATH.stat().st_size
        print(f"\nFAISS Index:")
        print(f"  Size: {idx_size / 1024:.1f} KB")
        print(f"  Path: {Config.FAISS_INDEX_PATH}")
    
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    """Clear cache and index."""
    Config.init_dirs()
    
    if args.yes or input("Clear all cached data? [y/N] ").lower() == "y":
        cache = CacheManager()
        count = cache.clear()
        print(f"Cleared {count} cached embeddings")
        
        if Config.FAISS_INDEX_PATH.exists():
            Config.FAISS_INDEX_PATH.unlink()
            print("Removed FAISS index")
        
        ids_path = Path(str(Config.FAISS_INDEX_PATH) + ".ids.npy")
        if ids_path.exists():
            ids_path.unlink()
    
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Document Embedding Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    idx_parser = subparsers.add_parser("index", help="Index documents")
    idx_parser.add_argument(
        "--data-dir", "-d",
        default=str(Config.DATA_DIR),
        help="Directory containing .txt files"
    )
    idx_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-embedding (ignore cache)"
    )
    idx_parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Use multiprocessing for loading"
    )
    
    srv_parser = subparsers.add_parser("serve", help="Start API server")
    srv_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    srv_parser.add_argument("--port", type=int, default=8000, help="Port number")
    srv_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="Results count")
    search_parser.add_argument("--expand", "-e", action="store_true", help="Expand query")
    
    subparsers.add_parser("stats", help="Show statistics")
    
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        "index": cmd_index,
        "serve": cmd_serve,
        "search": cmd_search,
        "stats": cmd_stats,
        "clear": cmd_clear,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())