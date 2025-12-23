"""
Unit tests for headers indexing strategy with mocks.

Verifies that header generation functions are called correctly
and headers are added to chunks before indexing.
"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.rag.techniques.indexing.headers import HeadersStrategy


@pytest.mark.unit
class TestHeadersIndexingUnit:
    """Unit tests for headers strategy with mocks."""

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.headers.generate_chunk_header")
    async def test_headers_generation_called_for_each_chunk(self, mock_generate_header):
        """Verify generate_chunk_header is called for each chunk."""
        # Setup mock to return predictable headers
        mock_generate_header.side_effect = [
            "Test Header 1",
            "Test Header 2",
            "Test Header 3",
        ]

        strategy = HeadersStrategy()
        chunks = [
            Document(page_content="Chunk 1 content", metadata={}),
            Document(page_content="Chunk 2 content", metadata={}),
            Document(page_content="Chunk 3 content", metadata={}),
        ]
        config = {
            "base_chunking": "standard",
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

        # Call post_process directly (this is where headers are generated)
        # We don't need to mock chunk() since we're testing post_process
        result = await strategy.post_process(chunks, config)

        # Verify header generation was called for each chunk
        assert mock_generate_header.call_count == len(chunks), (
            f"Expected {len(chunks)} calls to generate_chunk_header, "
            f"got {mock_generate_header.call_count}"
        )

        # Verify headers are in chunks
        assert result[0].metadata.get("header") == "Test Header 1"
        assert result[1].metadata.get("header") == "Test Header 2"
        assert result[2].metadata.get("header") == "Test Header 3"

        # Verify headers are prepended to page_content
        assert result[0].page_content.startswith("[Test Header 1]\n\n")
        assert result[1].page_content.startswith("[Test Header 2]\n\n")
        assert result[2].page_content.startswith("[Test Header 3]\n\n")

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.headers.generate_chunk_header")
    async def test_headers_format_correct(self, mock_generate_header):
        """Verify headers are in correct format: [header]\n\n{content}."""
        mock_generate_header.return_value = "Sample Header"

        strategy = HeadersStrategy()
        chunk = Document(page_content="Original content here", metadata={})
        config = {"base_chunking": "standard", "chunk_size": 100, "chunk_overlap": 20}

        result = await strategy.post_process([chunk], config)

        # Check format
        assert result[0].page_content == "[Sample Header]\n\nOriginal content here"
        assert result[0].metadata["header"] == "Sample Header"

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.headers.generate_chunk_header")
    async def test_headers_handles_generation_failure(self, mock_generate_header):
        """Verify that header generation failures don't crash the process."""
        # Mock to raise exception for some chunks
        mock_generate_header.side_effect = [
            "Header 1",
            Exception("LLM unavailable"),
            "Header 3",
        ]

        strategy = HeadersStrategy()
        chunks = [
            Document(page_content="Chunk 1", metadata={}),
            Document(page_content="Chunk 2", metadata={}),
            Document(page_content="Chunk 3", metadata={}),
        ]
        config = {"base_chunking": "standard", "chunk_size": 100, "chunk_overlap": 20}

        result = await strategy.post_process(chunks, config)

        # Should still process all chunks
        assert len(result) == 3

        # First and third should have headers
        assert result[0].metadata.get("header") == "Header 1"
        assert result[2].metadata.get("header") == "Header 3"

        # Second chunk should not have header (generation failed)
        assert "header" not in result[1].metadata
        # But original content should still be there
        assert result[1].page_content == "Chunk 2"

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.headers.generate_chunk_header")
    async def test_headers_with_standard_base_chunking(self, mock_generate_header):
        """Verify headers work with standard base chunking."""
        mock_generate_header.return_value = "Standard Base Header"

        strategy = HeadersStrategy()
        # Create mock documents as if from standard chunking
        chunks = [
            Document(page_content="Standard chunk content", metadata={"page": 1}),
        ]
        config = {
            "base_chunking": "standard",
            "chunk_size": 512,
            "chunk_overlap": 100,
        }

        # Test post_process directly (headers are added here)
        result = await strategy.post_process(chunks, config)

        # Verify header was generated
        assert mock_generate_header.called
        assert result[0].metadata.get("header") == "Standard Base Header"

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.headers.generate_chunk_header")
    async def test_headers_with_semantic_base_chunking(self, mock_generate_header):
        """Verify headers work with semantic base chunking."""
        mock_generate_header.return_value = "Semantic Base Header"

        strategy = HeadersStrategy()
        chunks = [
            Document(page_content="Semantic chunk content", metadata={"page": 1}),
        ]
        config = {
            "base_chunking": "semantic",
            "chunk_size": 1024,
            "chunk_overlap": 100,
        }

        # Test post_process directly
        result = await strategy.post_process(chunks, config)

        # Verify header was generated
        assert mock_generate_header.called
        assert result[0].metadata.get("header") == "Semantic Base Header"

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.headers.generate_chunk_header")
    async def test_headers_called_with_chunk_content(self, mock_generate_header):
        """Verify generate_chunk_header is called with correct chunk content."""
        mock_generate_header.return_value = "Test Header"

        strategy = HeadersStrategy()
        test_content = "This is the chunk content that should be passed to LLM"
        chunks = [Document(page_content=test_content, metadata={})]
        config = {"base_chunking": "standard", "chunk_size": 100, "chunk_overlap": 20}

        await strategy.post_process(chunks, config)

        # Verify the function was called with the chunk content
        mock_generate_header.assert_called_once_with(test_content)

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.headers.generate_chunk_header")
    async def test_headers_preserves_original_metadata(self, mock_generate_header):
        """Verify that original chunk metadata is preserved when adding headers."""
        mock_generate_header.return_value = "New Header"

        strategy = HeadersStrategy()
        original_metadata = {"page": 5, "source": "test.pdf", "line_start": 10}
        chunk = Document(page_content="Test content", metadata=original_metadata.copy())
        config = {"base_chunking": "standard", "chunk_size": 100, "chunk_overlap": 20}

        result = await strategy.post_process([chunk], config)

        # Verify original metadata is preserved
        for key, value in original_metadata.items():
            assert result[0].metadata[key] == value

        # Verify header is added
        assert result[0].metadata["header"] == "New Header"
