"""
Tests for standard indexing strategy.

Focuses on verifying that standard chunking works correctly:
- Chunks are created with correct sizes
- Metadata is properly set
- Chunks are indexed successfully
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
from tests.utils.verification import verify_standard_strategy


@pytest.mark.integration
@pytest.mark.standard
class TestStandardIndexing:
    """Test suite for standard indexing strategy."""

    @pytest.mark.parametrize("chunk_size", [512, 1024, 2048])
    @pytest.mark.parametrize("chunk_overlap", [100, 200])
    def test_standard_creates_chunks(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
        chunk_size,
        chunk_overlap,
    ):
        """Test that standard strategy creates chunks successfully."""
        # Upload document
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="standard",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        # Wait for processing
        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        # Verify document completed
        doc_details = get_document_details(api_client, document_id)
        assert doc_details["status"] == "completed", "Document should be completed"

        # Verify chunks were created
        chunks_data = get_chunks_from_vectorstore(document_id, sample_size=5)
        assert chunks_data["chunk_count"] > 0, "Should have created chunks"

    @pytest.mark.parametrize("chunk_size", [512, 1024, 2048])
    def test_standard_chunk_sizes(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
        chunk_size,
    ):
        """Test that standard chunks have correct sizes."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="standard",
            chunk_size=chunk_size,
            chunk_overlap=200,
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        # Get chunks and verify sizes
        chunks_data = get_chunks_from_vectorstore(document_id, sample_size=10)
        assert chunks_data["chunk_count"] > 0, "Should have created chunks"

        config = {"chunk_size": chunk_size, "chunk_overlap": 200}
        verification = verify_standard_strategy(chunks_data["chunk_samples"], config)

        assert verification["strategy_applied"], (
            f"Standard strategy verification failed: {verification.get('issues', [])}"
        )

    def test_standard_metadata(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
    ):
        """Test that standard chunks have proper metadata (line numbers)."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="standard",
            chunk_size=1024,
            chunk_overlap=200,
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
            # Standard strategy should add line_start and line_end
            assert "line_start" in metadata or "line_end" in metadata, (
                f"Chunk should have line number metadata. Got: {metadata.keys()}"
            )

    def test_standard_retrieval_smoke(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
    ):
        """Smoke test: verify standard chunks can be retrieved (basic functionality)."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="standard",
            chunk_size=1024,
            chunk_overlap=200,
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        # Basic retrieval test
        retrieval_result = perform_retrieval_test(
            api_client, document_id, query="What is the main topic?", top_k=5
        )

        assert retrieval_result["success"], (
            f"Retrieval should work: {retrieval_result.get('error', 'Unknown error')}"
        )
        assert retrieval_result["chunks_count"] > 0, "Should retrieve chunks"
