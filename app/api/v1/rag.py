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
from app.models.enums import RAGTechnique
from app.models.schemas import QueryRequest, QueryResponse
from app.rag.rules import TechniqueValidator
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


@router.post("/validate")
async def validate_techniques(techniques: List[RAGTechnique]):
    """Validate technique combination."""
    validator = TechniqueValidator()
    is_valid, errors, warnings = validator.validate(techniques)
    return {"valid": is_valid, "errors": errors, "warnings": warnings}
