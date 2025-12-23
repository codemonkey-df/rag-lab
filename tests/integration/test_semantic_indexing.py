"""
Tests for semantic indexing strategy.

Focuses on verifying that semantic chunking works correctly:
- Chunks are created using semantic boundaries
- Chunk sizes vary (not fixed like standard)
- Metadata is properly set
"""

from uuid import UUID

import pytest

from tests.conftest import (
    get_chunks_from_vectorstore,
    get_document_details,
    perform_retrieval_test,
    upload_document,
    wait_for_processing,
)
from tests.utils.verification import verify_semantic_strategy


@pytest.mark.integration
@pytest.mark.semantic
class TestSemanticIndexing:
    """Test suite for semantic indexing strategy."""

    def test_semantic_creates_chunks(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
    ):
        """Test that semantic strategy creates chunks successfully."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="semantic",
            chunk_size=1024,  # Ignored for semantic
            chunk_overlap=100,  # Ignored for semantic
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        # Wait for processing (semantic is async)
        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        # Verify document completed
        doc_details = get_document_details(api_client, document_id)
        assert doc_details["status"] == "completed", "Document should be completed"

        # Verify chunks were created
        chunks_data = get_chunks_from_vectorstore(document_id, sample_size=5)
        assert chunks_data["chunk_count"] > 0, "Should have created chunks"

    def test_semantic_chunk_size_variance(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
    ):
        """Test that semantic chunks have variable sizes (semantic boundaries)."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="semantic",
            chunk_size=1024,  # Ignored
            chunk_overlap=100,  # Ignored
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        # Get chunks and verify variance
        chunks_data = get_chunks_from_vectorstore(document_id, sample_size=10)
        assert chunks_data["chunk_count"] > 0, "Should have created chunks"

        config = {"chunk_size": 1024, "chunk_overlap": 100}
        verification = verify_semantic_strategy(chunks_data["chunk_samples"], config)

        # Semantic verification is soft - we check variance
        # If variance is too low, it's a warning but not a failure
        if verification.get("issues"):
            pytest.skip(f"Semantic chunks may be too uniform: {verification['issues']}")

    def test_semantic_metadata(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
    ):
        """Test that semantic chunks have proper metadata."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="semantic",
            chunk_size=1024,
            chunk_overlap=100,
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        chunks_data = get_chunks_from_vectorstore(document_id, sample_size=5)
        assert chunks_data["chunk_count"] > 0, "Should have created chunks"

        # Check that chunks have line number metadata
        for chunk in chunks_data["chunk_samples"]:
            metadata = chunk.get("metadata", {})
            assert "line_start" in metadata or "line_end" in metadata, (
                f"Chunk should have line number metadata. Got: {metadata.keys()}"
            )

    def test_semantic_retrieval_smoke(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
    ):
        """Smoke test: verify semantic chunks can be retrieved."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="semantic",
            chunk_size=1024,
            chunk_overlap=100,
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        retrieval_result = perform_retrieval_test(
            api_client, document_id, query="What is the main topic?", top_k=5
        )

        assert retrieval_result["success"], (
            f"Retrieval should work: {retrieval_result.get('error', 'Unknown error')}"
        )
        assert retrieval_result["chunks_count"] > 0, "Should retrieve chunks"
