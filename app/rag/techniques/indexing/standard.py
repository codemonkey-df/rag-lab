"""
Standard chunking indexing strategy using RecursiveCharacterTextSplitter.

This is the baseline indexing approach: simple, fast, and robust.
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.techniques.indexing.base import BaseIndexingStrategy
from app.services.vectorstore import get_chroma_collection

logger = logging.getLogger(__name__)


def add_line_numbers_to_chunks(chunks: List[Document]) -> List[Document]:
    """
    Add line_start and line_end metadata to chunks based on text position.

    Args:
        chunks: List of Document chunks

    Returns:
        List of Document chunks with line_start and line_end metadata
    """
    for chunk in chunks:
        if "line_start" not in chunk.metadata or "line_end" not in chunk.metadata:
            text = chunk.page_content
            # Estimate: ~80 characters per line (typical for PDFs)
            estimated_lines = max(1, len(text) // 80)

            if "line_start" not in chunk.metadata:
                chunk.metadata["line_start"] = 1
            if "line_end" not in chunk.metadata:
                chunk.metadata["line_end"] = (
                    chunk.metadata.get("line_start", 1) + estimated_lines
                )

    return chunks


class StandardStrategy(BaseIndexingStrategy):
    """
    Standard indexing strategy using RecursiveCharacterTextSplitter.

    Characteristics:
    - Fast: No LLM calls, pure text splitting
    - Predictable: Chunk sizes are controlled
    - Baseline: Good starting point for RAG systems
    - No post-processing: Just adds line numbers

    Performance:
    - 100+ page PDF: < 1 second
    - Runs synchronously (not in background)
    """

    async def chunk(
        self,
        documents: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Chunk documents using RecursiveCharacterTextSplitter.

        Args:
            documents: Raw documents from PDF loader
            config: Must contain 'chunk_size' and 'chunk_overlap'

        Returns:
            List of chunked documents with metadata
        """
        self.validate_config(config)

        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        chunks = splitter.split_documents(documents)
        logger.info(f"Standard chunking: {len(documents)} docs -> {len(chunks)} chunks")

        return chunks

    async def post_process(
        self,
        chunks: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Post-process: add line numbers to chunks.

        Args:
            chunks: Chunked documents
            config: Not used for standard strategy

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
        logger.info(f"Indexed {len(chunks)} chunks to ChromaDB for doc {document_id}")

    def supports_async_execution(self) -> bool:
        """Standard strategy is fast, runs synchronously."""
        return False
