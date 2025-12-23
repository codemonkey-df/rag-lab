"""
Unit tests for proposition indexing strategy with mocks.

Verifies that proposition generation and evaluation functions
are called correctly.
"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.rag.techniques.indexing.proposition import (
    PropositionStrategy,
)


@pytest.mark.unit
class TestPropositionIndexingUnit:
    """Unit tests for proposition strategy with mocks."""

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.proposition.generate_propositions")
    async def test_proposition_generation_called(self, mock_generate_propositions):
        """Verify generate_propositions is called for each chunk."""
        # Mock to return test propositions
        mock_generate_propositions.side_effect = [
            ["Proposition 1 from chunk 1", "Proposition 2 from chunk 1"],
            ["Proposition 1 from chunk 2"],
        ]

        strategy = PropositionStrategy()
        chunks = [
            Document(page_content="Chunk 1 content", metadata={"chunk_id": 1}),
            Document(page_content="Chunk 2 content", metadata={"chunk_id": 2}),
        ]
        config = {"chunk_size": 1024, "chunk_overlap": 50}

        result = await strategy.post_process(chunks, config)

        # Verify generate_propositions was called for each chunk
        assert mock_generate_propositions.call_count == len(chunks), (
            f"Expected {len(chunks)} calls to generate_propositions, "
            f"got {mock_generate_propositions.call_count}"
        )

        # Verify propositions were created
        assert len(result) == 3  # 2 + 1 propositions
        assert result[0].page_content == "Proposition 1 from chunk 1"
        assert result[1].page_content == "Proposition 2 from chunk 1"
        assert result[2].page_content == "Proposition 1 from chunk 2"

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.proposition.evaluate_proposition")
    @patch("app.rag.techniques.indexing.proposition.generate_propositions")
    async def test_proposition_quality_checking(
        self, mock_generate_propositions, mock_evaluate_proposition
    ):
        """Verify propositions are quality-checked."""
        # Mock proposition generation
        mock_generate_propositions.return_value = [
            "Good proposition",
            "Bad proposition",
        ]

        # Mock evaluation - first passes, second fails
        def mock_eval(prop, original):
            if "Good" in prop:
                return {
                    "accuracy": 8,
                    "clarity": 8,
                    "completeness": 8,
                    "conciseness": 8,
                }
            else:
                return {
                    "accuracy": 5,
                    "clarity": 5,
                    "completeness": 5,
                    "conciseness": 5,
                }

        mock_evaluate_proposition.side_effect = mock_eval

        strategy = PropositionStrategy()
        chunks = [
            Document(page_content="Original chunk", metadata={"chunk_id": 1}),
        ]
        config = {
            "chunk_size": 1024,
            "chunk_overlap": 50,
            "quality_thresholds": {
                "accuracy": 7,
                "clarity": 7,
                "completeness": 7,
                "conciseness": 7,
            },
        }

        result = await strategy.post_process(chunks, config)

        # Verify evaluation was called for each proposition
        assert mock_evaluate_proposition.call_count == 2

        # Only good proposition should pass (threshold is 7)
        assert len(result) == 1
        assert result[0].page_content == "Good proposition"

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.proposition.generate_propositions")
    async def test_proposition_handles_generation_failure(
        self, mock_generate_propositions
    ):
        """Verify that proposition generation failures don't crash."""
        # Mock to raise exception
        mock_generate_propositions.side_effect = [
            ["Proposition 1"],
            Exception("LLM unavailable"),
            ["Proposition 3"],
        ]

        strategy = PropositionStrategy()
        chunks = [
            Document(page_content="Chunk 1", metadata={"chunk_id": 1}),
            Document(page_content="Chunk 2", metadata={"chunk_id": 2}),
            Document(page_content="Chunk 3", metadata={"chunk_id": 3}),
        ]
        config = {"chunk_size": 1024, "chunk_overlap": 50}

        result = await strategy.post_process(chunks, config)

        # Should still process successful chunks
        assert len(result) >= 1
        assert result[0].page_content == "Proposition 1"

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.proposition.generate_propositions")
    async def test_proposition_preserves_metadata(self, mock_generate_propositions):
        """Verify that original chunk metadata is preserved in propositions."""
        mock_generate_propositions.return_value = ["Test proposition"]

        strategy = PropositionStrategy()
        original_metadata = {"chunk_id": 1, "page": 5, "source": "test.pdf"}
        chunk = Document(
            page_content="Original content", metadata=original_metadata.copy()
        )
        config = {"chunk_size": 1024, "chunk_overlap": 50}

        result = await strategy.post_process([chunk], config)

        # Verify metadata is preserved
        for key, value in original_metadata.items():
            assert result[0].metadata[key] == value

    @pytest.mark.asyncio
    @patch("app.rag.techniques.indexing.proposition.generate_propositions")
    async def test_proposition_called_with_chunk_content(
        self, mock_generate_propositions
    ):
        """Verify generate_propositions is called with chunk content."""
        mock_generate_propositions.return_value = ["Proposition"]

        strategy = PropositionStrategy()
        test_content = "This content should be passed to generate_propositions"
        chunks = [
            Document(page_content=test_content, metadata={"chunk_id": 1}),
        ]
        config = {"chunk_size": 1024, "chunk_overlap": 50}

        await strategy.post_process(chunks, config)

        # Verify called with chunk content
        mock_generate_propositions.assert_called_once_with(test_content)

    @pytest.mark.asyncio
    async def test_proposition_initial_chunking_uses_config(self):
        """Test that proposition strategy uses config values for chunking."""
        strategy = PropositionStrategy()
        documents = [
            Document(
                page_content="Word " * 1000,  # ~5000 chars
                metadata={"page": 1},
            ),
        ]
        config = {"chunk_size": 1024, "chunk_overlap": 50}

        result = await strategy.chunk(documents, config)

        # Verify chunks were created
        assert len(result) > 1, "Should create multiple chunks"
        assert all(isinstance(chunk, Document) for chunk in result)

        # Verify chunks have chunk_id metadata
        for i, chunk in enumerate(result):
            assert chunk.metadata.get("chunk_id") == i + 1

        # Verify chunk sizes are approximately correct (allowing for word boundary chunking)
        for chunk in result:
            chunk_length = len(chunk.page_content)
            # Allow wider range since chunking at word boundaries can create smaller chunks
            # For chunk_size=1024, chunks should be roughly 100-1200 (accounting for overlap and boundaries)
            assert 100 <= chunk_length <= 1200, (
                f"Chunk size {chunk_length} not in expected range 100-1200"
            )
