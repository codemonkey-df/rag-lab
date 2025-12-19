"""
Headers (contextual chunk headers) indexing strategy with LLM-generated titles.

This strategy preserves document hierarchy by generating contextual headers
(titles) for each chunk using an LLM. Headers are prepended to chunk content
to provide semantic context during retrieval.

Reference: https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/contextual_chunk_headers.ipynb
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.dependencies import get_llm
from app.rag.techniques.indexing.base import BaseIndexingStrategy
from app.rag.techniques.indexing.semantic import SemanticStrategy
from app.rag.techniques.indexing.standard import (
    StandardStrategy,
    add_line_numbers_to_chunks,
)
from app.services.vectorstore import get_chroma_collection

logger = logging.getLogger(__name__)


HEADER_PROMPT = """Generate a concise title (3-5 words) for the following text chunk.
The title should capture the main topic or theme of the chunk.

Text Chunk:
{chunk_text}

Title:"""


async def generate_chunk_header(chunk_text: str) -> str:
    """
    Generate a header/title for a chunk using LLM.

    Args:
        chunk_text: Text content of the chunk

    Returns:
        Generated header/title (3-5 words)
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(HEADER_PROMPT)
    chain = prompt | llm | StrOutputParser()

    header = await chain.ainvoke({"chunk_text": chunk_text})
    return header.strip()


class HeadersStrategy(BaseIndexingStrategy):
    """
    Headers (contextual chunk headers) indexing strategy.

    Combines a base chunking strategy (standard or semantic) with LLM-generated
    contextual headers to preserve document structure and hierarchy.

    Process:
    1. Chunk using base strategy (standard or semantic)
    2. Generate LLM-based header for each chunk
    3. Add header to chunk metadata
    4. Prepend header to chunk content
    5. Index chunks into vectorstore

    Characteristics:
    - Hierarchical: Preserves document structure via headers
    - LLM-enhanced: Headers capture semantic meaning
    - Flexible: Can use standard or semantic chunking as base
    - Very slow: LLM call for every chunk (20-page PDF ≈ 30 min)

    Performance:
    - 100+ page PDF: 30+ minutes (1 LLM call per chunk)
    - Runs as background task (async execution)
    - WARNING: Very slow on local LLMs

    Configuration:
    - chunk_size: Required (passed to base chunker)
    - chunk_overlap: Required (passed to base chunker)
    - base_chunking: "standard" or "semantic" (default: "standard")
    """

    async def chunk(
        self,
        documents: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Chunk documents using base strategy.

        Args:
            documents: Raw documents from PDF loader
            config: Must contain:
                   - chunk_size
                   - chunk_overlap
                   - base_chunking: "standard" or "semantic"

        Returns:
            List of chunked documents
        """
        self.validate_config(config)

        base_chunking = config.get("base_chunking", "standard")

        if base_chunking == "semantic":
            strategy = SemanticStrategy()
        else:
            strategy = StandardStrategy()

        chunks = await strategy.chunk(documents, config)
        logger.info(
            f"Headers strategy: used {base_chunking} base chunking -> {len(chunks)} chunks"
        )

        return chunks

    async def post_process(
        self,
        chunks: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Post-process: generate and add LLM-based headers to chunks.

        WARNING: This is very slow - one LLM call per chunk.

        Args:
            chunks: Chunked documents
            config: Optional progress_callback for tracking

        Returns:
            Chunks with headers added to metadata and content
        """
        total = len(chunks)
        progress_callback = config.get("progress_callback")

        for i, chunk in enumerate(chunks):
            try:
                # Generate header
                header = await generate_chunk_header(chunk.page_content)

                # Add to metadata
                chunk.metadata["header"] = header

                # Prepend header to content
                chunk.page_content = f"[{header}]\n\n{chunk.page_content}"

                # Update progress
                if progress_callback:
                    progress_callback((i + 1) / total * 100)

                logger.debug(f"Generated header: {header[:30]}...")

            except Exception as e:
                logger.warning(f"Failed to generate header for chunk {i}: {e}")
                # Continue with next chunk

        # Add line numbers
        chunks = add_line_numbers_to_chunks(chunks)

        logger.info(f"Added headers to {len(chunks)} chunks")
        return chunks

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
            f"Indexed {len(chunks)} header-enhanced chunks to ChromaDB for doc {document_id}"
        )

    def supports_async_execution(self) -> bool:
        """Headers strategy is very slow, must run as background task."""
        return True

    def get_required_config(self) -> List[str]:
        """Headers strategy requires base chunking config."""
        return ["chunk_size", "chunk_overlap", "base_chunking"]

    def get_optional_config(self) -> Dict[str, Any]:
        """Provide defaults for headers strategy."""
        return {
            "base_chunking": "standard",
            "progress_callback": None,
        }
