"""
FastAPI REST API for the embedding search engine.
Provides endpoints for semantic search, statistics, and index management.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .cache_manager import CacheManager
from .embedder import EmbeddingGenerator
from .preprocess import DocumentLoader, preprocess_documents
from .search_engine import QueryExpander, SearchEngine
from .utils import Config, get_logger

logger = get_logger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class SearchRequest(BaseModel):
    """Search request payload."""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="Search query text"
    )
    top_k: int = Field(
        default=5, 
        ge=1, 
        le=100, 
        description="Number of results to return"
    )
    expand_query: bool = Field(
        default=False, 
        description="Use WordNet query expansion"
    )
    length_normalize: bool = Field(
        default=False, 
        description="Apply document length normalization"
    )


class ExplanationResponse(BaseModel):
    """Ranking explanation details."""
    keywords: list[str]
    overlap_ratio: float
    doc_length: int
    raw_score: float | None = None
    length_normalized_score: float | None = None


class ResultItem(BaseModel):
    """Single search result."""
    doc_id: str
    score: float
    preview: str
    explanation: ExplanationResponse


class SearchResponse(BaseModel):
    """Search response payload."""
    results: list[ResultItem]
    query: str
    total_docs: int
    search_time_ms: float


class StatsResponse(BaseModel):
    """System statistics response."""
    indexed_documents: int
    cached_embeddings: int
    cache_db_size_bytes: int
    index_type: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    index_loaded: bool


class ReindexRequest(BaseModel):
    """Reindex request payload."""
    force: bool = Field(
        default=False, 
        description="Force re-embedding all documents"
    )


class DocumentInfo(BaseModel):
    """Document information."""
    doc_id: str
    filename: str
    length: int
    preview: str


class DocumentsResponse(BaseModel):
    """Documents list response."""
    documents: list[DocumentInfo]
    total: int


# =============================================================================
# Application State
# =============================================================================

class EngineState:
    """
    Application state container.
    
    Holds references to all engine components.
    """
    embedder: EmbeddingGenerator | None = None
    search_engine: SearchEngine | None = None
    cache: CacheManager | None = None
    query_expander: QueryExpander | None = None
    initialized: bool = False


# Global state instance
state = EngineState()


def initialize_engine(force_reindex: bool = False) -> None:
    """
    Initialize or reinitialize the search engine.
    
    Args:
        force_reindex: Force re-embedding all documents
    """
    # Ensure directories exist
    Config.init_dirs()
    
    # Initialize components
    state.cache = CacheManager()
    state.embedder = EmbeddingGenerator(cache_manager=state.cache)
    state.search_engine = SearchEngine(dimension=state.embedder.embedding_dim)
    state.query_expander = QueryExpander()
    
    # Load and index documents
    docs = preprocess_documents(Config.DATA_DIR)
    
    if docs:
        embeddings = state.embedder.embed_documents(docs, force=force_reindex)
        state.search_engine.index_documents(docs, embeddings)
        state.search_engine.save_index()
        logger.info(f"Indexed {len(docs)} documents")
    else:
        logger.warning("No documents found to index")
    
    state.initialized = True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    
    Initializes engine on startup, cleans up on shutdown.
    """
    logger.info("Initializing search engine...")
    initialize_engine()
    logger.info("Search engine ready")
    yield
    logger.info("Shutting down search engine")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Embedding Search API",
    description="Multi-document semantic search engine with intelligent caching",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check API health status.
    
    Returns:
        Health status with component states
    """
    return HealthResponse(
        status="healthy" if state.initialized else "initializing",
        model_loaded=state.embedder is not None,
        index_loaded=(
            state.search_engine is not None 
            and state.search_engine.document_count > 0
        )
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Perform semantic search.
    
    Steps:
    1. Optionally expand query with synonyms
    2. Embed the query
    3. Perform vector search
    4. Return ranked results with explanations
    
    Args:
        request: Search request with query and options
        
    Returns:
        Ranked results with explanations
    """
    if not state.initialized or not state.embedder or not state.search_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    query = request.query
    
    # Optional query expansion
    if request.expand_query and state.query_expander:
        query = state.query_expander.expand(query)
        logger.debug(f"Expanded query: {query}")
    
    # Embed query
    query_embedding = state.embedder.embed_query(query)
    
    # Perform search
    response = state.search_engine.search(
        query=request.query,  # Original query for display
        query_embedding=query_embedding,
        top_k=request.top_k,
        include_explanation=True,
        length_normalize=request.length_normalize
    )
    
    # Convert to response model
    results = [
        ResultItem(
            doc_id=r.doc_id,
            score=r.score,
            preview=r.preview,
            explanation=ExplanationResponse(**r.explanation)
        )
        for r in response.results
    ]
    
    return SearchResponse(
        results=results,
        query=response.query,
        total_docs=response.total_docs,
        search_time_ms=response.search_time_ms
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Get system statistics.
    
    Returns:
        Cache and index statistics
    """
    if not state.cache or not state.search_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    cache_stats = state.cache.get_stats()
    
    # Determine index type
    index_type = "faiss" if hasattr(state.search_engine.index, "index") else "numpy"
    
    return StatsResponse(
        indexed_documents=state.search_engine.document_count,
        cached_embeddings=cache_stats["total_entries"],
        cache_db_size_bytes=cache_stats["db_size_bytes"],
        index_type=index_type
    )


@app.get("/documents", response_model=DocumentsResponse)
async def list_documents() -> DocumentsResponse:
    """
    List all indexed documents.
    
    Returns:
        List of document info
    """
    if not state.search_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    docs = [
        DocumentInfo(
            doc_id=doc.doc_id,
            filename=doc.filename,
            length=doc.doc_length,
            preview=(
                doc.content[:100] + "..." 
                if len(doc.content) > 100 
                else doc.content
            )
        )
        for doc in state.search_engine.documents.values()
    ]
    
    return DocumentsResponse(documents=docs, total=len(docs))


@app.post("/reindex")
async def reindex(request: ReindexRequest) -> dict:
    """
    Trigger re-indexing of all documents.
    
    Args:
        request: Reindex options
        
    Returns:
        Reindex status
    """
    try:
        initialize_engine(force_reindex=request.force)
        return {
            "status": "success",
            "documents_indexed": (
                state.search_engine.document_count 
                if state.search_engine 
                else 0
            ),
            "force": request.force
        }
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)