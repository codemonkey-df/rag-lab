"""
Semantic chunking indexing strategy using embeddings-based text splitting.

Splits text based on semantic similarity between sentences, using embeddings
to detect natural breakpoints in meaning.

Reference: https://github.com/langchain-ai/langchain/blob/master/libs/experimental/langchain_experimental/text_splitter/semanticchunker.py
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from app.core.dependencies import get_embeddings
from app.rag.techniques.indexing.base import BaseIndexingStrategy
from app.rag.techniques.indexing.standard import add_line_numbers_to_chunks
from app.services.vectorstore import get_chroma_collection

logger = logging.getLogger(__name__)


class SemanticStrategy(BaseIndexingStrategy):
    """
    Semantic indexing strategy.

    Uses embedding-based text splitting to identify natural semantic boundaries
    rather than arbitrary character positions. Splits at points where the
    semantic similarity between sentences drops significantly.

    Characteristics:
    - Semantic awareness: Chunks at meaningful boundaries
    - Variable size: Chunk sizes vary based on semantic content
    - Slower: Requires embedding during ingestion
    - Better quality: More coherent chunks for semantic retrieval

    Performance:
    - 100+ page PDF: 10-30 seconds (includes embeddings)
    - Runs as background task (async execution)
    - Slower than standard, but better quality for semantic retrieval

    Configuration:
    - breakpoint_threshold_type: "percentile", "standard_deviation", or "interquartile"
    - breakpoint_threshold_amount: Threshold value (0-1 for percentile)
    """

    def __init__(self):
        """Initialize semantic strategy."""
        super().__init__()
        self._chunker = None

    def _get_semantic_chunker(
        self,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 0.95,
    ) -> SemanticChunker:
        """
        Create semantic chunker with configured parameters.

        Args:
            breakpoint_threshold_type: Type of threshold calculation
            breakpoint_threshold_amount: Threshold value

        Returns:
            SemanticChunker instance
        """
        embeddings = get_embeddings()

        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

    async def chunk(
        self,
        documents: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Chunk documents using semantic similarity.

        Args:
            documents: Raw documents from PDF loader
            config: Configuration with optional:
                   - chunk_size: Ignored for semantic chunking
                   - chunk_overlap: Ignored for semantic chunking
                   - breakpoint_threshold_type: Default "percentile"
                   - breakpoint_threshold_amount: Default 0.95

        Returns:
            List of semantically chunked documents
        """
        # Semantic chunking doesn't use chunk_size/overlap, but validate it anyway
        # for consistency with the interface
        if "chunk_size" not in config or "chunk_overlap" not in config:
            raise ValueError("Config must contain 'chunk_size' and 'chunk_overlap'")

        threshold_type = config.get("breakpoint_threshold_type", "percentile")
        threshold_amount = config.get("breakpoint_threshold_amount", 0.95)

        chunker = self._get_semantic_chunker(threshold_type, threshold_amount)
        chunks = chunker.split_documents(documents)

        logger.info(
            f"Semantic chunking: {len(documents)} docs -> {len(chunks)} chunks "
            f"(threshold: {threshold_type}={threshold_amount})"
        )

        return chunks

    async def post_process(
        self,
        chunks: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Post-process: add line numbers to semantic chunks.

        Args:
            chunks: Semantically chunked documents
            config: Not used

        Returns:
            Chunks with line number metadata
        """
        return add_line_numbers_to_chunks(chunks)

    async def index(
        self,
        document_id: UUID,
        chunks: List[Document],
    ) -> None:
        """
        Index chunks into ChromaDB.

        Args:
            document_id: Document UUID for collection name
            chunks: Documents to index

        Raises:
            Exception: If ChromaDB operation fails
        """
        collection = get_chroma_collection(document_id)
        collection.add_documents(chunks)
        logger.info(
            f"Indexed {len(chunks)} semantic chunks to ChromaDB for doc {document_id}"
        )

    def supports_async_execution(self) -> bool:
        """Semantic chunking is slow, should run as background task."""
        return True

    def get_optional_config(self) -> Dict[str, Any]:
        """Provide default semantic chunking parameters."""
        return {
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": 0.95,
        }
