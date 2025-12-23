"""
Integration tests for RAG query combinations with standard indexing.

Tests query, retrieval, and filtering techniques:
- basic_rag
- basic_rag + hyde
- basic_rag + reranking
- basic_rag + contextual_compression

All tests use the same document indexed with standard indexing strategy.
"""

import time
from typing import Any, Dict, List
from uuid import UUID

import pytest

from tests.conftest import (
    delete_document,
    get_document_details,
    upload_document,
    wait_for_processing,
)


@pytest.mark.integration
class TestRAGQueryCombinations:
    """Integration tests for RAG query combinations with standard indexing."""

    @pytest.fixture(scope="class")
    def indexed_document(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        server_health_check,
    ):
        """
        Upload and index document once for all tests using standard indexing.

        This fixture runs once per test class and provides the document_id
        to all test methods. Cleanup is handled in teardown.
        """
        # Upload document with standard indexing
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="standard",
            chunk_size=1024,
            chunk_overlap=200,
        )
        document_id = UUID(upload_result["document_id"])

        # Wait for processing to complete
        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        # Verify document is completed
        doc_details = get_document_details(api_client, document_id)
        assert doc_details["status"] == "completed", (
            f"Document should be completed, got status: {doc_details['status']}"
        )

        yield document_id

        # Cleanup: delete document after all tests in the class complete
        delete_document(api_client, document_id)

    def _execute_rag_query(
        self,
        api_client,
        document_id: UUID,
        techniques: List[str],
        query: str = "What is the main topic?",
        top_k: int = 5,
        temperature: float = 0.7,
        bm25_weight: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Execute RAG query with given technique combination.

        Args:
            api_client: httpx client for API requests
            document_id: UUID of the indexed document
            techniques: List of RAG technique strings
            query: Query string
            top_k: Number of chunks to retrieve
            temperature: LLM temperature
            bm25_weight: BM25 weight (for fusion retrieval, not used for basic_rag)

        Returns:
            Dictionary with query result or error information
        """
        payload = {
            "document_id": str(document_id),
            "query": query,
            "techniques": techniques,
            "query_params": {
                "top_k": top_k,
                "temperature": temperature,
                "bm25_weight": bm25_weight,
            },
        }

        try:
            start_time = time.time()
            response = api_client.post("/api/v1/rag/query", json=payload)
            elapsed = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "retrieved_chunks": result.get("retrieved_chunks", []),
                    "result_id": result.get("result_id"),
                    "scores": result.get("scores", {}),
                    "latency_ms": result.get("scores", {}).get("latency_ms", elapsed),
                    "chunks_count": len(result.get("retrieved_chunks", [])),
                }
            else:
                error_data = response.json() if response.content else {}
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "detail": error_data.get("detail", "Unknown error"),
                    "latency_ms": elapsed,
                }
        except Exception as e:
            return {
                "success": False,
                "error": "Request Error",
                "detail": str(e),
            }

    def test_basic_rag(self, api_client, indexed_document):
        """Test basic RAG retrieval without any enhancements."""
        result = self._execute_rag_query(
            api_client=api_client,
            document_id=indexed_document,
            techniques=["basic_rag"],
            query="What is the main topic?",
            top_k=5,
        )

        # Verify HTTP success
        assert result["success"], (
            f"Basic RAG query should succeed: {result.get('error', 'Unknown error')}"
        )

        # Verify response structure
        assert "response" in result, "Response should contain answer text"
        assert result["response"], "Response text should not be empty"
        assert "retrieved_chunks" in result, "Response should contain retrieved_chunks"
        assert "result_id" in result, "Response should contain result_id"
        assert "scores" in result, "Response should contain scores"

        # Verify chunks were retrieved
        assert result["chunks_count"] > 0, "Should retrieve at least one chunk"
        assert result["chunks_count"] <= 5, "Should not retrieve more than top_k chunks"

        # Verify chunk structure
        chunks = result["retrieved_chunks"]
        for chunk in chunks:
            assert "page" in chunk, "Each chunk should have page number"
            assert "text" in chunk, "Each chunk should have text content"
            assert chunk["text"], "Chunk text should not be empty"

        # Verify performance (basic RAG should be fast)
        assert result["latency_ms"] < 60000, (
            f"Basic RAG should complete in < 60s, took {result['latency_ms']:.0f}ms"
        )

    def test_basic_rag_with_hyde(self, api_client, indexed_document, require_llm):
        """Test basic RAG with HyDE query expansion."""
        result = self._execute_rag_query(
            api_client=api_client,
            document_id=indexed_document,
            techniques=["basic_rag", "hyde"],
            query="What is the main topic?",
            top_k=5,
        )

        # Verify HTTP success
        assert result["success"], (
            f"Basic RAG + HyDE query should succeed: {result.get('error', 'Unknown error')}"
        )

        # Verify response structure
        assert "response" in result, "Response should contain answer text"
        assert result["response"], "Response text should not be empty"
        assert "retrieved_chunks" in result, "Response should contain retrieved_chunks"
        assert "result_id" in result, "Response should contain result_id"

        # Verify chunks were retrieved
        assert result["chunks_count"] > 0, "Should retrieve at least one chunk"
        assert result["chunks_count"] <= 5, "Should not retrieve more than top_k chunks"

        # Verify chunk structure
        chunks = result["retrieved_chunks"]
        for chunk in chunks:
            assert "page" in chunk, "Each chunk should have page number"
            assert "text" in chunk, "Each chunk should have text content"
            assert chunk["text"], "Chunk text should not be empty"

        # Verify performance (HyDE adds query expansion overhead)
        assert result["latency_ms"] < 60000, (
            f"Basic RAG + HyDE should complete in < 60s, took {result['latency_ms']:.0f}ms"
        )

    def test_basic_rag_with_reranking(self, api_client, indexed_document):
        """Test basic RAG with reranking filter."""
        result = self._execute_rag_query(
            api_client=api_client,
            document_id=indexed_document,
            techniques=["basic_rag", "reranking"],
            query="What is the main topic?",
            top_k=5,
        )

        # Verify HTTP success
        assert result["success"], (
            f"Basic RAG + Reranking query should succeed: {result.get('error', 'Unknown error')}"
        )

        # Verify response structure
        assert "response" in result, "Response should contain answer text"
        assert result["response"], "Response text should not be empty"
        assert "retrieved_chunks" in result, "Response should contain retrieved_chunks"
        assert "result_id" in result, "Response should contain result_id"

        # Verify chunks were retrieved
        assert result["chunks_count"] > 0, "Should retrieve at least one chunk"
        assert result["chunks_count"] <= 5, "Should not retrieve more than top_k chunks"

        # Verify chunk structure
        chunks = result["retrieved_chunks"]
        for chunk in chunks:
            assert "page" in chunk, "Each chunk should have page number"
            assert "text" in chunk, "Each chunk should have text content"
            assert chunk["text"], "Chunk text should not be empty"

        # Verify performance (reranking adds cross-encoder overhead)
        assert result["latency_ms"] < 60000, (
            f"Basic RAG + Reranking should complete in < 60s, took {result['latency_ms']:.0f}ms"
        )

    def test_basic_rag_with_compression(
        self, api_client, indexed_document, require_llm
    ):
        """Test basic RAG with context compression filter."""
        result = self._execute_rag_query(
            api_client=api_client,
            document_id=indexed_document,
            techniques=["basic_rag", "contextual_compression"],
            query="What is the main topic?",
            top_k=5,
        )

        # Verify HTTP success
        assert result["success"], (
            f"Basic RAG + Compression query should succeed: {result.get('error', 'Unknown error')}"
        )

        # Verify response structure
        assert "response" in result, "Response should contain answer text"
        assert result["response"], "Response text should not be empty"
        assert "retrieved_chunks" in result, "Response should contain retrieved_chunks"
        assert "result_id" in result, "Response should contain result_id"

        # Verify chunks were retrieved
        assert result["chunks_count"] > 0, "Should retrieve at least one chunk"
        assert result["chunks_count"] <= 5, "Should not retrieve more than top_k chunks"

        # Verify chunk structure
        chunks = result["retrieved_chunks"]
        for chunk in chunks:
            assert "page" in chunk, "Each chunk should have page number"
            assert "text" in chunk, "Each chunk should have text content"
            assert chunk["text"], "Chunk text should not be empty"

        # Verify performance (compression adds LLM overhead)
        assert result["latency_ms"] < 60000, (
            f"Basic RAG + Compression should complete in < 60s, took {result['latency_ms']:.0f}ms"
        )
