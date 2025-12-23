"""
Unit tests for semantic indexing strategy with mocks.

Verifies that semantic chunker is used correctly.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.techniques.indexing.semantic import SemanticStrategy


@pytest.mark.unit
class TestSemanticIndexingUnit:
    """Unit tests for semantic indexing strategy with mocks."""

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.semantic.SemanticChunker")
    @patch("app.rag.techniques.indexing.semantic.get_embeddings")
    async def test_semantic_chunker_used(self, mock_get_embeddings, mock_chunker_class):
        """Verify SemanticChunker is used for chunking."""
        # Mock embeddings
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings

        # Mock chunker instance
        mock_chunker = MagicMock()
        mock_chunker.split_documents.return_value = [
            Document(page_content="Semantic chunk 1", metadata={}),
            Document(page_content="Semantic chunk 2", metadata={}),
        ]
        mock_chunker_class.return_value = mock_chunker

        strategy = SemanticStrategy()
        documents = [
            Document(page_content="Test document content", metadata={"page": 1}),
        ]
        config = {
            "chunk_size": 1024,  # Ignored
            "chunk_overlap": 100,  # Ignored
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": 0.95,
        }

        result = await strategy.chunk(documents, config)

        # Verify SemanticChunker was created with embeddings
        mock_chunker_class.assert_called_once()
        call_kwargs = mock_chunker_class.call_args[1]
        assert call_kwargs["embeddings"] == mock_embeddings
        assert call_kwargs["breakpoint_threshold_type"] == "percentile"
        assert call_kwargs["breakpoint_threshold_amount"] == 0.95

        # Verify split_documents was called
        mock_chunker.split_documents.assert_called_once_with(documents)

        # Verify result
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_semantic_post_process_adds_line_numbers(self):
        """Test that post_process adds line number metadata."""
        strategy = SemanticStrategy()
        chunks = [
            Document(page_content="Semantic chunk", metadata={}),
        ]
        config = {}

        result = await strategy.post_process(chunks, config)

        # Should have line_start and line_end
        assert "line_start" in result[0].metadata
        assert "line_end" in result[0].metadata

    def test_semantic_supports_async_execution(self):
        """Verify semantic strategy execution mode."""
        strategy = SemanticStrategy()

        # Semantic runs as background task (slow)
        assert strategy.supports_async_execution() is True

    def test_semantic_default_config(self):
        """Test semantic strategy default configuration."""
        strategy = SemanticStrategy()
        defaults = strategy.get_optional_config()

        assert defaults["breakpoint_threshold_type"] == "percentile"
        assert defaults["breakpoint_threshold_amount"] == 0.95
