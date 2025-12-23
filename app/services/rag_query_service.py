"""
RAG Query Service - Facade Pattern

Single entry point for all RAG query execution.
Orchestrates pipeline selection and execution based on technique configuration.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List
from uuid import UUID, uuid4

from sqlmodel import Session

from app.db.models import QueryResult
from app.db.repositories import QueryResultRepository
from app.models.enums import LAYER_3_TECHNIQUES, RAGTechnique
from app.models.schemas import ChunkInfo, QueryResponse, ComparisonResponse, ComparisonMetrics
from app.rag.pipelines.orchestration_pipeline import OrchestrationPipeline
from app.rag.pipelines.standard_pipeline import StandardRAGPipeline
from app.services.scoring import calculate_semantic_variance
from app.services.tracing import (
    capture_adaptive_metadata,
    capture_crag_metadata,
    capture_retrieved_chunks,
    capture_self_rag_metadata,
)

logger = logging.getLogger(__name__)


class RAGQueryService:
    """
    Facade for RAG query execution.

    Provides a unified interface for executing RAG queries with any combination
    of techniques. Handles pipeline selection (standard vs orchestration) and
    parameter management.
    """

    def __init__(self):
        """Initialize RAG query service."""
        pass

    async def execute_query(
        self,
        query: str,
        document_id: UUID,
        techniques: List[RAGTechnique],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a RAG query with specified techniques.

        Args:
            query: User query
            document_id: UUID of document to query
            techniques: List of RAGTechnique values
            **kwargs: Additional parameters (top_k, bm25_weight, temperature, etc.)

        Returns:
            Dictionary containing answer and metadata

        Raises:
            ValueError: If techniques are invalid or incompatible
        """
        logger.info(f"Executing RAG query with techniques: {techniques}")

        try:
            # Check for orchestration techniques (Layer 3)
            orchestration_techniques = [
                t for t in techniques if t in LAYER_3_TECHNIQUES
            ]

            if orchestration_techniques:
                # Use orchestration pipeline for Layer 3 techniques
                # Layer 3 techniques take over the entire pipeline
                technique = orchestration_techniques[0]  # Can only use one
                logger.info(f"Using orchestration pipeline with {technique}")

                pipeline = OrchestrationPipeline()
                result = await pipeline.execute(
                    query=query,
                    document_id=document_id,
                    technique=technique,
                    **kwargs,
                )
            else:
                # Use standard pipeline for Layer 1 and Layer 2 techniques
                logger.info("Using standard RAG pipeline")

                pipeline = StandardRAGPipeline()
                pipeline.build_stages(
                    techniques=techniques,
                    document_id=document_id,
                    top_k=kwargs.get("top_k", 5),
                    bm25_weight=kwargs.get("bm25_weight", 0.5),
                    temperature=kwargs.get("temperature", 0.7),
                    chunking_strategy=kwargs.get("chunking_strategy"),
                    chunk_size=kwargs.get("chunk_size"),
                    chunk_overlap=kwargs.get("chunk_overlap"),
                )

                result = await pipeline.execute(
                    query=query,
                    document_id=document_id,
                    top_k=kwargs.get("top_k", 5),
                    temperature=kwargs.get("temperature", 0.7),
                )

            logger.info("Query execution completed successfully")
            return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            raise

    async def execute_query_with_persistence(
        self,
        query: str,
        document_id: UUID,
        techniques: List[RAGTechnique],
        session_id: UUID,
        db_session: Session,
        **kwargs,
    ) -> QueryResponse:
        """
        Execute RAG query and handle all business logic.

        Includes:
        - Query execution
        - Result transformation and validation
        - Metadata capture (Layer 2 vs Layer 3)
        - Database persistence
        - Response formatting

        Args:
            query: User query
            document_id: UUID of document to query
            techniques: List of RAGTechnique values
            session_id: UUID of the session
            db_session: Database session
            **kwargs: Additional parameters (top_k, bm25_weight, temperature, etc.)

        Returns:
            QueryResponse ready to return from endpoint
        """
        start_time = time.time()

        try:
            # Execute query
            result_data = await self.execute_query(
                query=query,
                document_id=document_id,
                techniques=techniques,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Validate result_data
            if result_data is None:
                logger.error("Query service returned None")
                return QueryResponse(
                    response="Query execution returned no result. Please try again.",
                    retrieved_chunks=[],
                    result_id=uuid4(),
                    scores={"latency_ms": latency_ms, "token_count_est": 0},
                )

            if not isinstance(result_data, dict):
                logger.error(
                    f"Query service returned unexpected type: {type(result_data)}"
                )
                result_data = {"answer": str(result_data), "documents": []}

            # Extract response and documents
            response = result_data.get("answer", "")
            documents = result_data.get("documents", [])

            # Validate response is not empty
            if not response:
                logger.warning("Query service returned empty response")
                response = "I couldn't generate a response. Please try rephrasing your question."

            # Capture chunks based on technique type
            is_layer3 = any(t in LAYER_3_TECHNIQUES for t in techniques)

            if is_layer3:
                # Capture Layer 3 metadata with error handling
                try:
                    if RAGTechnique.SELF_RAG in techniques:
                        metadata = capture_self_rag_metadata(result_data)
                    elif RAGTechnique.CRAG in techniques:
                        metadata = capture_crag_metadata(result_data)
                    elif RAGTechnique.ADAPTIVE_RETRIEVAL in techniques:
                        metadata = capture_adaptive_metadata(result_data)
                    else:
                        metadata = {"chunks": []}
                except Exception as e:
                    logger.error(
                        f"Error capturing Layer 3 metadata: {e}", exc_info=True
                    )
                    metadata = {"chunks": []}

                retrieved_chunks_json = json.dumps(metadata.get("chunks", []))
                retrieved_chunks_response = [
                    ChunkInfo(
                        page=chunk.get("page", "?"),
                        text=chunk.get("text", ""),
                        line_start=chunk.get("line_start"),
                        line_end=chunk.get("line_end"),
                        score=chunk.get("score"),
                    )
                    for chunk in metadata.get("chunks", [])
                ]
            else:
                # Capture chunks for Layer 2 using tracing service with error handling
                try:
                    chunks_data = capture_retrieved_chunks(documents)
                except Exception as e:
                    logger.error(f"Error capturing chunks: {e}", exc_info=True)
                    chunks_data = []

                retrieved_chunks_json = json.dumps(chunks_data)
                retrieved_chunks_response = [
                    ChunkInfo(
                        page=chunk.get("page", "?"),
                        text=chunk.get("text", ""),
                        line_start=chunk.get("line_start"),
                        line_end=chunk.get("line_end"),
                        score=chunk.get("score"),
                    )
                    for chunk in chunks_data
                ]

            # Calculate metrics
            token_count_est = len(query + response) // 4

            # Store result with error handling
            try:
                repo = QueryResultRepository(db_session)
                result = QueryResult(
                    id=uuid4(),
                    session_id=session_id,
                    document_id=document_id,
                    query=query,
                    response=response,
                    latency_ms=latency_ms,
                    token_count_est=token_count_est,
                    techniques_used=json.dumps([t.value for t in techniques]),
                    query_params=json.dumps(kwargs),
                    retrieved_chunks=retrieved_chunks_json,
                )
                result = repo.create(result)
                result_id = result.id
            except Exception as e:
                logger.error(f"Error storing result in database: {e}", exc_info=True)
                # Still return response even if storage fails
                result_id = uuid4()

            return QueryResponse(
                response=response,
                retrieved_chunks=retrieved_chunks_response,
                result_id=result_id,
                scores={
                    "latency_ms": latency_ms,
                    "token_count_est": token_count_est,
                },
            )

        except Exception as e:
            logger.error(f"Error in execute_query_with_persistence: {e}", exc_info=True)
            latency_ms = (time.time() - start_time) * 1000
            return QueryResponse(
                response=f"Failed to execute query: {str(e)}. Please check your technique selection.",
                retrieved_chunks=[],
                result_id=uuid4(),
                scores={"latency_ms": latency_ms, "token_count_est": 0},
            )

    async def execute_parallel_comparison(
        self,
        query: str,
        document_id: UUID,
        techniques_1: List[RAGTechnique],
        techniques_2: List[RAGTechnique],
        session_id: UUID,
        db_session: Session,
        query_params_1: Dict[str, Any] = None,
        query_params_2: Dict[str, Any] = None,
    ) -> ComparisonResponse:
        """
        Execute two RAG queries in parallel with different techniques and return comparison.

        Args:
            query: User query (same for both pipelines)
            document_id: UUID of document to query
            techniques_1: List of RAGTechnique values for pipeline 1
            techniques_2: List of RAGTechnique values for pipeline 2
            session_id: UUID of the session
            db_session: Database session
            query_params_1: Query parameters for pipeline 1
            query_params_2: Query parameters for pipeline 2

        Returns:
            ComparisonResponse with results from both pipelines and metrics
        """
        logger.info("Starting parallel query comparison")

        if query_params_1 is None:
            query_params_1 = {}
        if query_params_2 is None:
            query_params_2 = {}

        try:
            # Execute both queries in parallel
            result_1_task = self.execute_query_with_persistence(
                query=query,
                document_id=document_id,
                techniques=techniques_1,
                session_id=session_id,
                db_session=db_session,
                **query_params_1,
            )

            result_2_task = self.execute_query_with_persistence(
                query=query,
                document_id=document_id,
                techniques=techniques_2,
                session_id=session_id,
                db_session=db_session,
                **query_params_2,
            )

            # Execute in parallel
            result_1, result_2 = await asyncio.gather(result_1_task, result_2_task)

            # Calculate semantic similarity
            try:
                semantic_similarity = await calculate_semantic_variance(
                    result_1.response, result_2.response
                )
            except Exception as e:
                logger.error(f"Error calculating semantic variance: {e}", exc_info=True)
                semantic_similarity = 0.5  # Default middle value if calculation fails

            # Calculate latency difference
            latency_1 = result_1.scores.get("latency_ms", 0)
            latency_2 = result_2.scores.get("latency_ms", 0)
            latency_diff_ms = abs(latency_1 - latency_2)

            # Interpret similarity
            if semantic_similarity > 0.8:
                interpretation = "Very similar"
            elif semantic_similarity > 0.6:
                interpretation = "Similar"
            elif semantic_similarity > 0.4:
                interpretation = "Different"
            else:
                interpretation = "Very different"

            # Build comparison metrics
            comparison = ComparisonMetrics(
                semantic_similarity=float(semantic_similarity),
                latency_diff_ms=latency_diff_ms,
                interpretation=interpretation,
            )

            logger.info("Parallel query comparison completed successfully")

            return ComparisonResponse(
                pipeline_1_result=result_1,
                pipeline_2_result=result_2,
                comparison=comparison,
            )

        except Exception as e:
            logger.error(f"Error in execute_parallel_comparison: {e}", exc_info=True)
            # Return error response with empty results
            error_response = QueryResponse(
                response=f"Comparison failed: {str(e)}",
                retrieved_chunks=[],
                result_id=uuid4(),
                scores={"latency_ms": 0, "token_count_est": 0},
            )
            return ComparisonResponse(
                pipeline_1_result=error_response,
                pipeline_2_result=error_response,
                comparison=ComparisonMetrics(
                    semantic_similarity=0.0,
                    latency_diff_ms=0.0,
                    interpretation="Comparison failed",
                ),
            )


# Singleton instance
_service = None


def get_rag_query_service() -> RAGQueryService:
    """
    Get or create singleton RAGQueryService instance.

    Returns:
        RAGQueryService instance
    """
    global _service
    if _service is None:
        _service = RAGQueryService()
    return _service
