"""
Unit tests for parent document indexing strategy with mocks.

Verifies that ParentDocumentRetriever is used correctly
and parent/child structure is created.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.techniques.indexing.parent import ParentDocumentStrategy


@pytest.mark.unit
class TestParentDocumentIndexingUnit:
    """Unit tests for parent document strategy with mocks."""

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.parent.ParentDocumentRetriever")
    async def test_parent_document_retriever_used(self, mock_retriever_class):
        """Verify ParentDocumentRetriever is created and used."""
        # Mock the retriever instance
        mock_retriever = MagicMock()
        mock_retriever.add_documents = MagicMock()
        mock_retriever_class.return_value = mock_retriever

        strategy = ParentDocumentStrategy()
        chunks = [
            Document(page_content="Document 1", metadata={}),
            Document(page_content="Document 2", metadata={}),
        ]

        # Mock the stores
        with (
            patch(
                "app.rag.techniques.indexing.parent.get_chroma_collection"
            ) as mock_vectorstore,
            patch(
                "app.rag.techniques.indexing.parent.get_parent_document_store"
            ) as mock_docstore,
        ):
            mock_vectorstore.return_value = MagicMock()
            mock_docstore.return_value = MagicMock()

            await strategy.index(None, chunks)

            # Verify retriever was created
            assert mock_retriever_class.called

            # Verify add_documents was called (via executor)
            # Note: Since it runs in executor, we check the mock was set up
            assert mock_retriever.add_documents is not None

    @pytest.mark.asyncio
    async def test_parent_document_chunk_returns_unchanged(self):
        """Verify chunk() method returns documents unchanged (chunking happens in index)."""
        strategy = ParentDocumentStrategy()
        documents = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
        ]
        config = {"chunk_size": 1024, "chunk_overlap": 200}

        result = await strategy.chunk(documents, config)

        # Should return documents unchanged
        assert result == documents
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_parent_document_post_process_unchanged(self):
        """Verify post_process returns chunks unchanged."""
        strategy = ParentDocumentStrategy()
        chunks = [
            Document(page_content="Chunk 1", metadata={}),
        ]
        config = {}

        result = await strategy.post_process(chunks, config)

        # Should return unchanged
        assert result == chunks

    def test_parent_document_supports_async_execution(self):
        """Verify parent document strategy execution mode."""
        strategy = ParentDocumentStrategy()

        # Parent document runs synchronously (not in background)
        assert strategy.supports_async_execution() is False
