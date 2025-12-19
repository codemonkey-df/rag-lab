"""
Contextual Compression for retrieved documents
"""

import asyncio
import logging
from typing import List

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.dependencies import get_llm
from app.rag.techniques.filtering.base import BaseFiltering

logger = logging.getLogger(__name__)

# Minimum number of chunks to keep even if compression filters them out
MIN_CHUNKS = 1


class SafeCompressionRetriever(BaseRetriever):
    """
    Compression retriever with safety checks to ensure minimum chunks.

    Wraps ContextualCompressionRetriever to prevent filtering out all chunks.
    """

    def __init__(
        self, base_retriever: BaseRetriever, compressor, min_chunks: int = MIN_CHUNKS
    ):
        """
        Initialize safe compression retriever.

        Args:
            base_retriever: Base retriever to wrap
            compressor: Document compressor
            min_chunks: Minimum number of chunks to keep
        """
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation for internal attributes
        object.__setattr__(
            self,
            "compression_retriever",
            ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
            ),
        )
        object.__setattr__(self, "base_retriever", base_retriever)
        object.__setattr__(self, "min_chunks", min_chunks)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve and compress documents with safety check.

        Args:
            query: Query string

        Returns:
            List of compressed documents (minimum MIN_CHUNKS)
        """
        try:
            # Try compression first
            compressed_docs = self.compression_retriever.invoke(query)

            # Check if we have minimum chunks
            if len(compressed_docs) >= self.min_chunks:
                logger.info(f"Compression kept {len(compressed_docs)} chunks")
                return compressed_docs
            else:
                logger.warning(
                    f"Compression returned only {len(compressed_docs)} chunks, "
                    f"falling back to uncompressed (need minimum {self.min_chunks})"
                )
                # Fall back to uncompressed documents
                original_docs = self.base_retriever.invoke(query)
                return original_docs[
                    : self.min_chunks * 2
                ]  # Return more uncompressed docs
        except Exception as e:
            logger.error(
                f"Compression failed: {e}, falling back to uncompressed", exc_info=True
            )
            # Fall back to uncompressed on error
            try:
                original_docs = self.base_retriever.invoke(query)
                return original_docs
            except Exception as e2:
                logger.error(f"Base retriever also failed: {e2}", exc_info=True)
                return []

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async version of retrieve and compress with safety check and parallel processing.

        This method retrieves documents first, then compresses them in parallel
        for significant performance improvement over sequential compression.

        Args:
            query: Query string

        Returns:
            List of compressed documents (minimum MIN_CHUNKS)
        """
        try:
            # Step 1: Retrieve documents first (fast)
            original_docs = await self.base_retriever.ainvoke(query)

            if not original_docs:
                logger.warning("No documents retrieved from base retriever")
                return []

            logger.info(
                f"Retrieved {len(original_docs)} documents, starting parallel compression"
            )

            # Step 2: Compress all documents in parallel
            async def compress_single_doc(
                doc: Document, index: int
            ) -> tuple[int, Document]:
                """Compress a single document and return it with its index."""
                try:
                    # Use the compressor directly
                    compressor = self.compression_retriever.base_compressor
                    compressed = await compressor.acompress_documents([doc], query)
                    if compressed:
                        logger.debug(
                            f"Compressed doc {index}: {len(doc.page_content)} -> {len(compressed[0].page_content)} chars"
                        )
                        return (index, compressed[0])
                    else:
                        logger.warning(
                            f"Compression returned empty for doc {index}, keeping original"
                        )
                        return (index, doc)
                except Exception as e:
                    logger.warning(
                        f"Compression failed for doc {index}: {e}, keeping original"
                    )
                    return (index, doc)

            # Create tasks for all documents
            tasks = [compress_single_doc(doc, i) for i, doc in enumerate(original_docs)]

            # Execute all compressions in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results: filter exceptions and sort by original index
            compressed_docs = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Compression task failed with exception: {result}")
                    # Skip this document
                elif isinstance(result, tuple):
                    index, doc = result
                    compressed_docs.append((index, doc))

            # Sort by index to maintain order and extract documents
            compressed_docs.sort(key=lambda x: x[0])
            final_docs = [doc for _, doc in compressed_docs]

            logger.info(f"Parallel compression completed: {len(final_docs)} documents")

            # Check if we have minimum chunks
            if len(final_docs) >= self.min_chunks:
                logger.info(f"Compression kept {len(final_docs)} chunks")
                return final_docs
            else:
                logger.warning(
                    f"Compression returned only {len(final_docs)} chunks, "
                    f"falling back to uncompressed (need minimum {self.min_chunks})"
                )
                # Fall back to uncompressed documents
                return original_docs[
                    : self.min_chunks * 2
                ]  # Return more uncompressed docs

        except Exception as e:
            logger.error(
                f"Parallel compression failed: {e}, falling back to uncompressed",
                exc_info=True,
            )
            # Fall back to uncompressed on error
            try:
                original_docs = await self.base_retriever.ainvoke(query)
                return original_docs
            except Exception as e2:
                logger.error(f"Base retriever also failed: {e2}", exc_info=True)
                return []


class CompressionFilter(BaseFiltering):
    """Compression filter that compresses documents using LLM."""

    def __init__(self, **kwargs):
        """
        Initialize compression filter.

        Args:
            **kwargs: Additional arguments
        """
        pass

    async def filter(
        self,
        documents: List[Document],
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Document]:
        """
        Compress documents to extract relevant information.

        Args:
            documents: List of documents to compress
            query: Query string for context
            top_k: Maximum number of documents to compress
            **kwargs: Additional parameters

        Returns:
            List of compressed documents
        """
        return await compress_documents(documents, query, top_k)


def create_compressor():
    """
    Create LLM-based document compressor.

    Returns:
        LLMChainExtractor instance configured with LLM
    """
    llm = get_llm()
    compressor = LLMChainExtractor.from_llm(llm)
    return compressor


async def compress_documents(
    documents: List[Document], query: str, top_k: int = 5
) -> List[Document]:
    """
    Standalone compression function for post-reranking compression.

    This function compresses only the top_k documents in parallel.
    It should be called AFTER reranking, not as a retriever wrapper.

    This is the recommended approach for the funnel strategy:
    Retrieve (K=25) → Rerank (to top 5) → Compress (only top 3-5)

    Args:
        documents: List of documents to compress
        query: Query string for context
        top_k: Maximum number of documents to compress (default: 5)

    Returns:
        List of compressed documents (up to top_k)
    """
    if not documents:
        logger.warning("No documents provided for compression")
        return []

    # Limit to top_k documents
    docs_to_compress = documents[:top_k]

    logger.info(f"Starting parallel compression of {len(docs_to_compress)} documents")

    try:
        # Create compressor
        compressor = create_compressor()

        # Compress each document in parallel
        async def compress_single_doc(
            doc: Document, index: int
        ) -> tuple[int, Document]:
            """Compress a single document and return it with its index."""
            try:
                compressed = await compressor.acompress_documents([doc], query)
                if compressed:
                    original_len = len(doc.page_content)
                    compressed_len = len(compressed[0].page_content)
                    logger.debug(
                        f"Compressed doc {index}: {original_len} -> {compressed_len} chars ({compressed_len * 100 // original_len}%)"
                    )
                    return (index, compressed[0])
                else:
                    logger.warning(
                        f"Compression returned empty for doc {index}, keeping original"
                    )
                    return (index, doc)
            except Exception as e:
                logger.warning(
                    f"Compression failed for doc {index}: {e}, keeping original"
                )
                return (index, doc)

        # Create tasks for all documents
        tasks = [compress_single_doc(doc, i) for i, doc in enumerate(docs_to_compress)]

        # Execute all compressions in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results: filter exceptions and sort by original index
        compressed_docs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Compression task failed with exception: {result}")
                # Skip this document
            elif isinstance(result, tuple):
                index, doc = result
                compressed_docs.append((index, doc))

        # Sort by index to maintain order and extract documents
        compressed_docs.sort(key=lambda x: x[0])
        final_docs = [doc for _, doc in compressed_docs]

        logger.info(
            f"Parallel compression completed: {len(final_docs)}/{len(docs_to_compress)} documents"
        )

        return final_docs

    except Exception as e:
        logger.error(
            f"Parallel compression failed: {e}, returning original documents",
            exc_info=True,
        )
        return docs_to_compress


def apply_compression(retriever: BaseRetriever) -> BaseRetriever:
    """
    Apply contextual compression to a retriever with safety checks.

    DEPRECATED: This approach compresses documents during retrieval (before reranking).
    For better performance, use compress_documents() AFTER reranking instead.

    Compresses retrieved documents to only relevant parts using LLM,
    but ensures minimum number of chunks are kept.

    Args:
        retriever: Base retriever to wrap with compression

    Returns:
        SafeCompressionRetriever wrapping the base retriever
    """
    compressor = create_compressor()
    compression_retriever = SafeCompressionRetriever(
        base_retriever=retriever, compressor=compressor, min_chunks=MIN_CHUNKS
    )
    return compression_retriever
