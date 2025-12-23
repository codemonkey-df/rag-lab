"""
Pydantic schemas for API requests and responses
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel

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


class PipelineConfig(BaseModel):
    """Configuration for a RAG pipeline."""

    techniques: List[RAGTechnique]
    query_params: Dict[str, Any] = {}  # top_k, bm25_weight, temperature


class ComparisonRequest(BaseModel):
    """Request schema for parallel pipeline comparison."""

    document_id: UUID
    query: str  # Same query for both pipelines
    pipeline_1: PipelineConfig
    pipeline_2: PipelineConfig
    session_id: Optional[UUID] = None


class ComparisonMetrics(BaseModel):
    """Metrics comparing two query results."""

    semantic_similarity: float  # 0.0 to 1.0
    latency_diff_ms: float  # Difference in latency between pipelines
    interpretation: str  # "Very similar", "Similar", "Different", "Very different"


class ComparisonResponse(BaseModel):
    """Response schema for pipeline comparison."""

    pipeline_1_result: QueryResponse
    pipeline_2_result: QueryResponse
    comparison: ComparisonMetrics


# RAG Configuration Schemas
class TechniqueInfo(BaseModel):
    """Information about a RAG technique."""

    value: str
    label: str
    description: str
    required: bool = False
    mutually_exclusive: bool = False
    mutually_exclusive_with: List[str] = []
    requires: List[str] = []
    enables: List[str] = []
    warning: Optional[str] = None
    default: bool = False


class LayerValidationRule(BaseModel):
    """Validation rules for a layer."""

    selection_type: str  # "single_required", "single_optional", "multi_optional"
    mutually_exclusive: bool = False
    conflicts: List[Dict[str, Any]] = []
    dependencies: List[Dict[str, Any]] = []


class RAGDefaults(BaseModel):
    """Default configuration values."""

    indexing_strategy: str
    techniques: List[str]
    chunk_size: int
    chunk_overlap: int


class RAGConfigurationResponse(BaseModel):
    """Complete RAG configuration response."""

    indexing_strategies: Dict[str, Dict[str, Any]]
    techniques: Dict[str, List[TechniqueInfo]]
    validation_rules: Dict[str, LayerValidationRule]
    defaults: RAGDefaults
