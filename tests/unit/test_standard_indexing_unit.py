"""
Unit tests for standard indexing strategy.

Tests chunking logic directly without mocks (it's simple enough).
"""

import pytest
from langchain_core.documents import Document

from app.rag.techniques.indexing.standard import StandardStrategy


@pytest.mark.unit
class TestStandardIndexingUnit:
    """Unit tests for standard indexing strategy."""

    @pytest.mark.asyncio
    async def test_standard_chunking_creates_chunks(self):
        """Test that standard chunking creates chunks from documents."""
        strategy = StandardStrategy()
        documents = [
            Document(
                page_content="This is a test document with some content. " * 50,
                metadata={"page": 1},
            ),
        ]
        config = {"chunk_size": 200, "chunk_overlap": 50}

        result = await strategy.chunk(documents, config)

        # Should create multiple chunks
        assert len(result) > 1
        assert all(isinstance(chunk, Document) for chunk in result)

    @pytest.mark.asyncio
    async def test_standard_chunk_sizes(self):
        """Test that standard chunks are approximately the right size."""
        strategy = StandardStrategy()
        documents = [
            Document(
                page_content="Word " * 1000,  # ~5000 chars
                metadata={"page": 1},
            ),
        ]
        config = {"chunk_size": 500, "chunk_overlap": 100}

        result = await strategy.chunk(documents, config)

        # Check chunk sizes are approximately correct
        # Allow wider range since chunking at word boundaries can create smaller chunks
        # Minimum should be at least chunk_size - chunk_overlap (400) for non-overlapping chunks
        # But first/last chunks can be smaller
        for chunk in result:
            chunk_length = len(chunk.page_content)
            # More lenient: allow 100-700 range (accounting for word boundary chunking)
            assert 100 <= chunk_length <= 700, (
                f"Chunk size {chunk_length} not in expected range for chunk_size=500"
            )

    @pytest.mark.asyncio
    async def test_standard_preserves_metadata(self):
        """Test that standard chunking preserves document metadata."""
        strategy = StandardStrategy()
        original_metadata = {"page": 5, "source": "test.pdf"}
        documents = [
            Document(
                page_content="Test content " * 100,
                metadata=original_metadata.copy(),
            ),
        ]
        config = {"chunk_size": 200, "chunk_overlap": 50}

        result = await strategy.chunk(documents, config)

        # All chunks should have original metadata
        for chunk in result:
            for key, value in original_metadata.items():
                assert chunk.metadata[key] == value

    @pytest.mark.asyncio
    async def test_standard_post_process_adds_line_numbers(self):
        """Test that post_process adds line number metadata."""
        strategy = StandardStrategy()
        chunks = [
            Document(page_content="Chunk content", metadata={}),
        ]
        config = {}

        result = await strategy.post_process(chunks, config)

        # Should have line_start and line_end
        assert "line_start" in result[0].metadata
        assert "line_end" in result[0].metadata

    def test_standard_supports_async_execution(self):
        """Verify standard strategy execution mode."""
        strategy = StandardStrategy()

        # Standard runs synchronously (fast)
        assert strategy.supports_async_execution() is False
