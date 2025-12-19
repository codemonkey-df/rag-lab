"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel
from uuid import UUID
from typing import List, Optional, Dict, Any
from app.models.enums import RAGTechnique


class QueryRequest(BaseModel):
    """Request schema for RAG query."""
    document_id: UUID
    query: str
    techniques: List[RAGTechnique]
    query_params: Dict[str, Any] = {}  # top_k, bm25_weight, temperature
    session_id: Optional[UUID] = None


class ChunkInfo(BaseModel):
    """Information about a retrieved chunk."""
    page: int | str
    text: str
    line_start: int | None = None
    line_end: int | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    """Response schema for RAG query."""
    response: str
    retrieved_chunks: List[ChunkInfo]
    result_id: UUID
    scores: Dict[str, Any]  # latency_ms, token_count_est, etc.
