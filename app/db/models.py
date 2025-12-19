"""
SQLModel database models
"""
from sqlmodel import SQLModel, Field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional


class Session(SQLModel, table=True):
    """Session model for grouping documents and queries"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)


class Document(SQLModel, table=True):
    """Document model for uploaded PDFs"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    filename: str
    file_path: str  # Stored file location
    status: str  # pending, processing, completed, failed
    indexing_progress: int = Field(default=0)  # 0-100% for UI Progress Bars
    chunking_strategy: str  # standard, parent_document, semantic, proposition, headers
    chunk_size: int  # 256/512/1024/2048 - index-time parameter
    chunk_overlap: int = Field(default=0)  # index-time parameter
    chroma_collection: str  # ChromaDB collection name
    embedding_dimension: int = Field(default=768)  # Nomic default
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None


class QueryResult(SQLModel, table=True):
    """Query result model with advanced scoring"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    document_id: UUID = Field(foreign_key="document.id")
    query: str
    response: str  # Generated answer
    # Metrics (inline calculation)
    latency_ms: float
    token_count_est: int  # Estimate: len(prompt) / 4
    semantic_variance: Optional[float] = None  # Score (0-1) comparing similarity to baseline
    # JSON Blobs
    techniques_used: str  # JSON list of Layer 2 & 3 techniques
    query_params: str  # JSON: top_k, bm25_weight, temperature, etc.
    retrieved_chunks: str  # JSON with page/line metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
