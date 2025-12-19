"""
Documents API endpoints
"""

import os
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlmodel import Session

from app.core.config import get_settings
from app.core.dependencies import get_embedding_dimension
from app.db.database import get_session
from app.db.models import Document
from app.db.models import Session as SessionModel
from app.db.repositories import DocumentRepository, SessionRepository
from app.services.bm25_manager import bm25_manager
from app.services.ingestion import DocumentIngestionService
from app.services.vectorstore import delete_collection_for_document

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
settings = get_settings()
ingestion_service = DocumentIngestionService()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    chunking_strategy: str = "standard",
    base_chunking: str = "standard",  # For headers strategy
    session_id: UUID | None = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db_session: Session = Depends(get_session),
):
    """
    Upload PDF and ingest with specified strategy.

    This endpoint is now a thin controller that:
    1. Validates the file
    2. Manages session and document records
    3. Delegates all ingestion logic to DocumentIngestionService

    All processing logic has been moved to the ingestion service,
    making this endpoint focused solely on HTTP concerns.
    """

    # Validate file
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Validate chunking strategy
    available_strategies = list(ingestion_service.get_available_strategies().keys())
    if chunking_strategy not in available_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chunking strategy: {chunking_strategy}. "
            f"Available: {', '.join(available_strategies)}",
        )

    # Create or get session
    if not session_id:
        session_id = uuid4()
        session_repo = SessionRepository(db_session)
        session = SessionModel(id=session_id)
        session_repo.create(session)
    else:
        session_repo = SessionRepository(db_session)
        session = session_repo.get_by_id(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

    # Save file
    doc_id = uuid4()
    upload_dir = Path(settings.upload_dir) / str(doc_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create document record
    repo = DocumentRepository(db_session)
    document = Document(
        id=doc_id,
        session_id=session_id,
        filename=file.filename,
        file_path=str(file_path),
        status="pending",
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chroma_collection=f"doc_{doc_id}",
        embedding_dimension=get_embedding_dimension(),
    )
    document = repo.create(document)

    # Build configuration for the strategy
    config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "base_chunking": base_chunking,  # For headers strategy
    }

    # Delegate to ingestion service
    try:
        result = await ingestion_service.ingest_document(
            document_id=doc_id,
            file_path=str(file_path),
            strategy_name=chunking_strategy,
            config=config,
            db_session=db_session,
            background_tasks=background_tasks,
        )
        return {
            "document_id": result["document_id"],
            "status": result["status"],
            "session_id": session_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("")
async def list_documents(
    session_id: UUID | None = None,
    db_session: Session = Depends(get_session),
):
    """List documents, optionally filtered by session."""
    repo = DocumentRepository(db_session)
    if session_id:
        return repo.list_by_session(session_id)
    return repo.list_all()


@router.get("/{doc_id}")
async def get_document(
    doc_id: UUID,
    db_session: Session = Depends(get_session),
):
    """Get document details."""
    repo = DocumentRepository(db_session)
    doc = repo.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: UUID,
    db_session: Session = Depends(get_session),
):
    """Delete document, Chroma collection, and BM25 index."""
    repo = DocumentRepository(db_session)
    doc = repo.get_by_id(doc_id)
    if not doc: 
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete Chroma collection
    delete_collection_for_document(doc_id)

    # Clear BM25 cache
    bm25_manager.clear_cache(doc_id)

    # Delete file
    if os.path.exists(doc.file_path):
        # Delete the file
        os.remove(doc.file_path)
        # Delete the directory if empty
        file_dir = Path(doc.file_path).parent
        if file_dir.exists() and not any(file_dir.iterdir()):
            file_dir.rmdir()

    # Delete database record
    repo.delete(doc_id)

    return {"message": "Document deleted"}


@router.get("/strategies/available")
async def get_available_strategies():
    """Get list of available indexing strategies with their metadata."""
    strategies = ingestion_service.get_available_strategies()
    return {
        "strategies": strategies,
        "count": len(strategies),
    }
