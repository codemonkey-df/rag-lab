"""
RAG query API endpoints - Thin Controller Pattern

Endpoints delegate all business logic to RAGQueryService.
Controllers are responsible only for:
- HTTP validation and marshalling
- Session management
- Lock management
- Error handling (HTTP level)
"""

import logging
from typing import List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.core.concurrency import llm_lock_manager
from app.db.database import get_session
from app.db.models import Session as SessionModel
from app.db.repositories import DocumentRepository, SessionRepository
from app.models.enums import (
    LAYER_1_TECHNIQUES,
    LAYER_2_TECHNIQUES,
    LAYER_3_TECHNIQUES,
    RAGTechnique,
)
from app.models.schemas import (
    ComparisonRequest,
    ComparisonResponse,
    LayerValidationRule,
    QueryRequest,
    QueryResponse,
    RAGConfigurationResponse,
    RAGDefaults,
    TechniqueInfo,
)
from app.rag.rules import TechniqueValidator
from app.services.ingestion import DocumentIngestionService
from app.services.rag_query_service import get_rag_query_service

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    db_session: Session = Depends(get_session),
):
    """Execute RAG query - thin controller delegating to service.

    This endpoint handles only HTTP-level concerns:
    - Technique validation
    - Lock management (prevent concurrent queries during indexing)
    - Session management
    - Service delegation

    All business logic (document fetching, result transformation,
    metadata capture, DB persistence) is handled by RAGQueryService.
    """

    # Validate techniques
    validator = TechniqueValidator()
    is_valid, errors, warnings = validator.validate(request.techniques)
    if not is_valid:
        raise HTTPException(
            status_code=400, detail={"errors": errors, "warnings": warnings}
        )

    # Check lock (can't query while indexing)
    can_query = await llm_lock_manager.acquire_for_query()
    if not can_query:
        raise HTTPException(
            status_code=503, detail="Service Busy: Indexing in progress..."
        )

    try:
        # Create or get session
        if not request.session_id:
            request.session_id = uuid4()
            session_repo = SessionRepository(db_session)
            session = SessionModel(id=request.session_id)
            session_repo.create(session)

        # Validate document is ready (complements concurrency lock)
        doc_repo = DocumentRepository(db_session)
        doc = doc_repo.get_by_id(request.document_id)
        if not doc:
            raise HTTPException(
                status_code=404, detail=f"Document {request.document_id} not found"
            )
        if doc.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Document {request.document_id} is not ready for querying. "
                f"Current status: {doc.status}. Please wait for indexing to complete.",
            )

        # Delegate ALL business logic to service
        rag_service = get_rag_query_service()
        response = await rag_service.execute_query_with_persistence(
            query=request.query,
            document_id=request.document_id,
            techniques=request.techniques,
            session_id=request.session_id,
            db_session=db_session,
            top_k=request.query_params.get("top_k", 5),
            bm25_weight=request.query_params.get("bm25_weight", 0.5),
            temperature=request.query_params.get("temperature", 0.7),
        )

        return response

    except HTTPException:
        # Re-raise HTTPExceptions (validation errors, lock errors)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await llm_lock_manager.release_for_query()


@router.post("/compare", response_model=ComparisonResponse)
async def compare_pipelines(
    request: ComparisonRequest,
    db_session: Session = Depends(get_session),
):
    """Execute two RAG pipelines in parallel and compare results.

    This endpoint handles HTTP-level concerns:
    - Technique validation for both pipelines
    - Lock management
    - Session management
    - Service delegation for parallel execution

    The service executes both queries concurrently and returns comparison metrics.
    """

    # Validate both pipeline configurations
    validator = TechniqueValidator()
    is_valid_1, errors_1, warnings_1 = validator.validate(request.pipeline_1.techniques)
    is_valid_2, errors_2, warnings_2 = validator.validate(request.pipeline_2.techniques)

    if not is_valid_1 or not is_valid_2:
        raise HTTPException(
            status_code=400,
            detail={
                "pipeline_1": {"errors": errors_1, "warnings": warnings_1}
                if not is_valid_1
                else None,
                "pipeline_2": {"errors": errors_2, "warnings": warnings_2}
                if not is_valid_2
                else None,
            },
        )

    # Check lock (can't query while indexing)
    can_query = await llm_lock_manager.acquire_for_query()
    if not can_query:
        raise HTTPException(
            status_code=503, detail="Service Busy: Indexing in progress..."
        )

    try:
        # Create or get session
        if not request.session_id:
            request.session_id = uuid4()
            session_repo = SessionRepository(db_session)
            session = SessionModel(id=request.session_id)
            session_repo.create(session)

        # Validate document is ready
        doc_repo = DocumentRepository(db_session)
        doc = doc_repo.get_by_id(request.document_id)
        if not doc:
            raise HTTPException(
                status_code=404, detail=f"Document {request.document_id} not found"
            )
        if doc.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Document {request.document_id} is not ready for querying. "
                f"Current status: {doc.status}. Please wait for indexing to complete.",
            )

        # Delegate to service for parallel execution
        rag_service = get_rag_query_service()
        response = await rag_service.execute_parallel_comparison(
            query=request.query,
            document_id=request.document_id,
            techniques_1=request.pipeline_1.techniques,
            techniques_2=request.pipeline_2.techniques,
            session_id=request.session_id,
            db_session=db_session,
            query_params_1=request.pipeline_1.query_params,
            query_params_2=request.pipeline_2.query_params,
        )

        return response

    except HTTPException:
        # Re-raise HTTPExceptions (validation errors, lock errors)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in compare endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await llm_lock_manager.release_for_query()


@router.get("/configuration", response_model=RAGConfigurationResponse)
async def get_rag_configuration():
    """Get complete RAG configuration for frontend.

    Returns all available techniques organized by layer,
    indexing strategies, validation rules, and defaults.
    This endpoint provides the single source of truth for
    RAG configuration that the frontend uses.
    """
    ingestion_service = DocumentIngestionService()

    # Get indexing strategies
    indexing_strategies = ingestion_service.get_available_strategies()

    # Build technique configuration
    def build_layer_config(techniques_set, layer_num):
        """Build configuration for a layer."""
        techniques_list = []
        for technique in techniques_set:
            # Map technique to display info
            technique_info = {
                "value": technique.value,
                "label": technique.value.replace("_", " ").title(),
                "description": _get_technique_description(technique),
                "required": layer_num == 1,
                "mutually_exclusive": layer_num in (1, 3),
                "mutually_exclusive_with": _get_mutually_exclusive_with(technique),
                "requires": _get_technique_requires(technique),
                "enables": _get_technique_enables(technique),
                "warning": _get_technique_warning(technique),
                "default": technique == RAGTechnique.STANDARD_CHUNKING,
            }
            techniques_list.append(TechniqueInfo(**technique_info))
        return techniques_list

    techniques = {
        "layer_1": build_layer_config(LAYER_1_TECHNIQUES, 1),
        "layer_2": build_layer_config(LAYER_2_TECHNIQUES, 2),
        "layer_3": build_layer_config(LAYER_3_TECHNIQUES, 3),
    }

    # Build validation rules
    validation_rules = {
        "layer_1": LayerValidationRule(
            selection_type="single_required",
            mutually_exclusive=True,
            conflicts=[
                {
                    "techniques": ["semantic_chunking", "proposition_chunking"],
                    "reason": "Different storage structures",
                },
                {
                    "techniques": ["proposition_chunking", "contextual_headers"],
                    "reason": "Incompatible",
                },
            ],
        ),
        "layer_2": LayerValidationRule(
            selection_type="multi_optional",
            mutually_exclusive=False,
            conflicts=[
                {
                    "techniques": ["basic_rag", "fusion_retrieval"],
                    "reason": "Fusion contains Basic",
                },
            ],
            dependencies=[
                {
                    "technique": "reranking",
                    "requires": ["basic_rag", "fusion_retrieval"],
                },
                {
                    "technique": "contextual_compression",
                    "requires": ["basic_rag", "fusion_retrieval"],
                },
            ],
        ),
        "layer_3": LayerValidationRule(
            selection_type="single_optional",
            mutually_exclusive=True,
            conflicts=[
                {
                    "techniques": ["self_rag", "crag"],
                    "reason": "Logic collision - both have critique loops",
                },
            ],
        ),
    }

    # Defaults
    defaults = RAGDefaults(
        indexing_strategy="standard",
        techniques=["standard_chunking", "basic_rag"],
        chunk_size=1024,
        chunk_overlap=200,
    )

    return RAGConfigurationResponse(
        indexing_strategies=indexing_strategies,
        techniques=techniques,
        validation_rules=validation_rules,
        defaults=defaults,
    )


def _get_technique_description(technique: RAGTechnique) -> str:
    """Get description for a technique."""
    descriptions = {
        RAGTechnique.STANDARD_CHUNKING: "Fast baseline fixed-size chunking",
        RAGTechnique.PARENT_DOCUMENT: "Larger parent chunks with smaller child retrieval",
        RAGTechnique.SEMANTIC_CHUNKING: "Embedding-based semantic sentence chunking",
        RAGTechnique.CONTEXTUAL_HEADERS: "Headers + standard/semantic chunking with context",
        RAGTechnique.PROPOSITION_CHUNKING: "Atomic fact extraction with LLM",
        RAGTechnique.HYDE: "Hypothetical Document Embeddings for query expansion",
        RAGTechnique.BASIC_RAG: "Standard semantic search retrieval",
        RAGTechnique.FUSION_RETRIEVAL: "Hybrid lexical + semantic search",
        RAGTechnique.RERANKING: "Rerank results for improved relevance",
        RAGTechnique.CONTEXTUAL_COMPRESSION: "Compress retrieved context",
        RAGTechnique.SELF_RAG: "Self-reflective RAG with relevance checking",
        RAGTechnique.CRAG: "Corrective RAG with knowledge validation",
        RAGTechnique.ADAPTIVE_RETRIEVAL: "Adaptive routing between retrieval methods",
    }
    return descriptions.get(technique, "")


def _get_technique_warning(technique: RAGTechnique) -> str | None:
    """Get warning for a technique."""
    warnings = {
        RAGTechnique.SEMANTIC_CHUNKING: "Slower than standard (embeds during split)",
        RAGTechnique.CONTEXTUAL_HEADERS: "Very slow (~30 min for 20-page PDF on local LLM)",
        RAGTechnique.PROPOSITION_CHUNKING: "Extremely slow (LLM rewrites every sentence)",
        RAGTechnique.SELF_RAG: "3x latency (3 LLM calls per query)",
        RAGTechnique.CRAG: "Requires web search, may increase latency",
    }
    return warnings.get(technique)


def _get_mutually_exclusive_with(technique: RAGTechnique) -> List[str]:
    """Get techniques that are mutually exclusive with this one."""
    exclusions = {
        RAGTechnique.BASIC_RAG: ["fusion_retrieval"],
        RAGTechnique.FUSION_RETRIEVAL: ["basic_rag"],
        RAGTechnique.SEMANTIC_CHUNKING: ["proposition_chunking"],
        RAGTechnique.PROPOSITION_CHUNKING: ["semantic_chunking", "contextual_headers"],
        RAGTechnique.CONTEXTUAL_HEADERS: ["proposition_chunking"],
        RAGTechnique.SELF_RAG: ["crag"],
        RAGTechnique.CRAG: ["self_rag"],
    }
    return exclusions.get(technique, [])


def _get_technique_requires(technique: RAGTechnique) -> List[str]:
    """Get techniques that this one requires."""
    requires = {
        RAGTechnique.RERANKING: ["basic_rag", "fusion_retrieval"],
        RAGTechnique.CONTEXTUAL_COMPRESSION: ["basic_rag", "fusion_retrieval"],
    }
    return requires.get(technique, [])


def _get_technique_enables(technique: RAGTechnique) -> List[str]:
    """Get techniques that this one enables."""
    enables = {
        RAGTechnique.BASIC_RAG: ["reranking", "contextual_compression"],
        RAGTechnique.FUSION_RETRIEVAL: ["reranking", "contextual_compression"],
    }
    return enables.get(technique, [])


@router.post("/validate")
async def validate_techniques(techniques: List[RAGTechnique]):
    """Validate technique combination."""
    validator = TechniqueValidator()
    is_valid, errors, warnings = validator.validate(techniques)
    return {"valid": is_valid, "errors": errors, "warnings": warnings}
