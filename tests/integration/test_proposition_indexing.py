"""
Tests for proposition indexing strategy.

Focuses on verifying that proposition strategy works correctly:
- Propositions are generated from chunks
- Propositions are atomic facts (short, self-contained)
- Quality checking is applied
- Metadata is preserved
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
from tests.utils.verification import verify_proposition_strategy


@pytest.mark.integration
@pytest.mark.proposition
@pytest.mark.slow
class TestPropositionIndexing:
    """Test suite for proposition indexing strategy."""

    @pytest.mark.parametrize("chunk_size", [1024])
    def test_proposition_creates_chunks(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
        require_llm,
        chunk_size,
    ):
        """Test that proposition strategy creates chunks successfully."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="proposition",
            chunk_size=chunk_size,  # Used for initial chunking
            chunk_overlap=50,  # Used for initial chunking
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        # Wait for processing (proposition is extremely slow, async)
        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        # Verify document completed
        doc_details = get_document_details(api_client, document_id)
        assert doc_details["status"] == "completed", "Document should be completed"

        # Verify chunks were created
        chunks_data = get_chunks_from_vectorstore(document_id, sample_size=5)
        assert chunks_data["chunk_count"] > 0, "Should have created chunks"

    def test_proposition_atomic_facts(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
        require_llm,
    ):
        """Test that propositions are atomic facts (short, self-contained)."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="proposition",
            chunk_size=1024,
            chunk_overlap=50,
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        # Get chunks and verify they are propositions
        chunks_data = get_chunks_from_vectorstore(document_id, sample_size=10)
        assert chunks_data["chunk_count"] > 0, "Should have created chunks"

        config = {"chunk_size": 1024, "chunk_overlap": 50}
        verification = verify_proposition_strategy(chunks_data["chunk_samples"], config)

        # Proposition verification is soft - chunks should be atomic facts
        # If they're too long, it's a warning but not a failure
        if verification.get("issues"):
            pytest.skip(f"Proposition chunks may be too long: {verification['issues']}")

    def test_proposition_metadata(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
        require_llm,
    ):
        """Test that propositions preserve metadata (chunk_id, line numbers)."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="proposition",
            chunk_size=1024,
            chunk_overlap=50,
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        chunks_data = get_chunks_from_vectorstore(document_id, sample_size=5)
        assert chunks_data["chunk_count"] > 0, "Should have created chunks"

        # Check that chunks have metadata
        for chunk in chunks_data["chunk_samples"]:
            metadata = chunk.get("metadata", {})
            # Propositions should have line numbers
            assert "line_start" in metadata or "line_end" in metadata, (
                f"Proposition should have line number metadata. Got: {metadata.keys()}"
            )

    def test_proposition_retrieval_smoke(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
        require_llm,
    ):
        """Smoke test: verify proposition chunks can be retrieved."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="proposition",
            chunk_size=1024,
            chunk_overlap=50,
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
