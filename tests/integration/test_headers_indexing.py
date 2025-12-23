"""
Tests for headers (contextual chunk headers) indexing strategy.

Focuses on verifying that headers strategy works correctly:
- Headers are generated and added to chunks
- Headers are in correct format: [header]\n\n{content}
- Headers are stored in metadata
- Works with both standard and semantic base chunking
"""

from uuid import UUID

import pytest

from tests.conftest import (
    get_chunks_from_vectorstore,
    perform_retrieval_test,
    upload_document,
    wait_for_processing,
)
from tests.utils.verification import verify_headers_strategy


@pytest.mark.integration
@pytest.mark.headers
@pytest.mark.slow
class TestHeadersIndexing:
    """Test suite for headers indexing strategy."""

    @pytest.mark.parametrize("base_chunking", ["standard", "semantic"])
    def test_headers_format(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
        require_llm,
        base_chunking,
    ):
        """Test that headers are in correct format: [header]\n\n{content}."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="headers",
            chunk_size=1024,
            chunk_overlap=100,
            base_chunking=base_chunking,
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        # Get chunks and verify headers
        chunks_data = get_chunks_from_vectorstore(document_id, sample_size=15)
        assert chunks_data["chunk_count"] > 0, "Should have created chunks"

        config = {
            "chunk_size": 1024,
            "chunk_overlap": 100,
            "base_chunking": base_chunking,
        }
        verification = verify_headers_strategy(chunks_data["chunk_samples"], config)

        assert verification["strategy_applied"], (
            f"Headers strategy verification failed: {verification.get('issues', [])}. "
            f"Expected headers in format [header]\\n\\n{{content}} or in metadata['header']"
        )

        # If we found evidence, log it
        if verification.get("evidence"):
            print(f"\nHeaders found: {verification['evidence']}")

    def test_headers_metadata(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
        require_llm,
    ):
        """Test that headers are stored in chunk metadata."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="headers",
            chunk_size=1024,
            chunk_overlap=200,
            base_chunking="standard",
        )
        document_id = UUID(upload_result["document_id"])
        cleanup_document(document_id)

        if upload_result.get("status") == "processing":
            wait_for_processing(api_client, document_id, max_wait=max_wait_time)

        chunks_data = get_chunks_from_vectorstore(document_id, sample_size=10)
        assert chunks_data["chunk_count"] > 0, "Should have created chunks"

        # Check that at least some chunks have header in metadata
        chunks_with_header_metadata = 0
        for chunk in chunks_data["chunk_samples"]:
            metadata = chunk.get("metadata", {})
            if metadata.get("header"):
                chunks_with_header_metadata += 1

        # At least some chunks should have header in metadata
        # (not all may have it if header generation failed for some)
        assert chunks_with_header_metadata > 0, (
            f"Expected at least some chunks to have header in metadata. "
            f"Checked {len(chunks_data['chunk_samples'])} chunks."
        )

    def test_headers_retrieval_smoke(
        self,
        api_client,
        test_pdf_path,
        max_wait_time,
        cleanup_document,
        server_health_check,
        require_llm,
    ):
        """Smoke test: verify headers chunks can be retrieved."""
        upload_result = upload_document(
            api_client=api_client,
            file_path=test_pdf_path,
            strategy="headers",
            chunk_size=1024,
            chunk_overlap=200,
            base_chunking="standard",
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
