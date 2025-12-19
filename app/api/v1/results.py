"""
Results management API endpoints
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session
from uuid import UUID
from typing import List
from app.db.database import get_session
from app.db.repositories import QueryResultRepository
from app.services.scoring import calculate_semantic_variance, calculate_batch_semantic_variance
import json

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/results", tags=["results"])


@router.get("/{session_id}")
async def get_results(
    session_id: UUID,
    db_session: Session = Depends(get_session),
):
    """Get all results for a session."""
    repo = QueryResultRepository(db_session)
    return repo.list_by_session(session_id)


@router.get("/compare")
async def compare_results(
    result_ids: str = Query(..., description="Comma-separated result IDs"),
    db_session: Session = Depends(get_session),
):
    """Compare multiple results side-by-side with semantic variance."""
    repo = QueryResultRepository(db_session)
    
    # Parse IDs
    ids = [UUID(id.strip()) for id in result_ids.split(",")]
    
    # Get results
    results = [repo.get_by_id(id) for id in ids]
    results = [r for r in results if r is not None]
    
    if len(results) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 results to compare")
    
    # Calculate semantic variance matrix for all responses
    responses = [r.response for r in results]
    similarity_matrix = await calculate_batch_semantic_variance(responses)
    
    # Build pairwise comparisons with interpretation labels
    comparisons = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            variance = float(similarity_matrix[i][j])
            
            # Add interpretation
            if variance > 0.8:
                interpretation = "Very similar"
            elif variance > 0.6:
                interpretation = "Similar"
            elif variance > 0.4:
                interpretation = "Different"
            else:
                interpretation = "Very different"
            
            comparisons.append({
                "result_id_1": str(results[i].id),
                "result_id_2": str(results[j].id),
                "semantic_variance": variance,
                "interpretation": interpretation,
            })
    
    # Helper function to safely parse JSON fields
    def safe_json_loads(json_str: str | None, default=None):
        """Safely parse JSON string, returning default if parsing fails."""
        if json_str is None or json_str == "":
            return default if default is not None else []
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse JSON field: {e}, value: {json_str[:100] if json_str else None}")
            return default if default is not None else []
    
    return {
        "results": [
            {
                "id": str(r.id), 
                "query": r.query, 
                "response": r.response,
                "techniques": safe_json_loads(r.techniques_used, []),
                "scores": {
                    "latency_ms": r.latency_ms,
                    "token_count_est": r.token_count_est,
                }
            } 
            for r in results
        ],
        "comparisons": comparisons,
        "similarity_matrix": similarity_matrix.tolist(),
    }


@router.get("/detail/{result_id}")
async def get_result_detail(
    result_id: UUID,
    db_session: Session = Depends(get_session),
):
    """Get single result details."""
    repo = QueryResultRepository(db_session)
    result = repo.get_by_id(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result
