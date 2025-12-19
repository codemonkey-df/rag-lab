"""
Reranking implementation using CrossEncoder
"""

import asyncio
import logging
from typing import List, Optional

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.rag.techniques.filtering.base import BaseFiltering

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """
    Detect available device for CrossEncoder inference.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Mac Metal Performance Shaders
    else:
        return "cpu"


class RerankingFilter(BaseFiltering):
    """Reranks documents using cross-encoder with device acceleration."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize reranking filter.

        Args:
            model_name: CrossEncoder model name
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            **kwargs: Additional arguments
        """
        if device is None:
            device = detect_device()

        self.device = device
        self.model = CrossEncoder(model_name, device=device)
        logger.info(f"Reranker initialized with device: {device}")

    async def filter(
        self,
        documents: List[Document],
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Document]:
        """
        Rerank and filter documents using cross-encoder.

        Args:
            documents: List of documents to rerank
            query: Query text
            top_k: Number of top documents to return
            **kwargs: Additional parameters

        Returns:
            Reranked list of documents
        """
        if not documents:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_filter, documents, query, top_k
        )

    def _sync_filter(
        self, documents: List[Document], query: str, top_k: int
    ) -> List[Document]:
        """
        Synchronous reranking implementation.

        Args:
            documents: List of documents to rerank
            query: Query text
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents
        """
        # Prepare pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get scores (uses device automatically via CrossEncoder)
        scores = self.model.predict(pairs)

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Update metadata with reranking score
        for doc, score in scored_docs[:top_k]:
            doc.metadata["rerank_score"] = float(score)

        # Return top k
        return [doc for doc, score in scored_docs[:top_k]]


class Reranker:
    """
    DEPRECATED: Use RerankingFilter instead.

    Reranks documents using cross-encoder with device acceleration.
    This class is kept for backward compatibility.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize reranker with CrossEncoder model.

        Args:
            model_name: CrossEncoder model name
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        if device is None:
            device = detect_device()

        self.device = device
        self.model = CrossEncoder(model_name, device=device)
        logger.info(f"Reranker initialized with device: {device}")

    def rerank(
        self, documents: List[Document], query: str, top_k: int = 5
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder.

        Args:
            documents: List of documents to rerank
            query: Query text
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents
        """
        if not documents:
            return []

        # Prepare pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get scores (uses device automatically via CrossEncoder)
        scores = self.model.predict(pairs)

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Update metadata with reranking score
        for doc, score in scored_docs[:top_k]:
            doc.metadata["rerank_score"] = float(score)

        # Return top k
        return [doc for doc, score in scored_docs[:top_k]]

    async def arerank(
        self, documents: List[Document], query: str, top_k: int = 5
    ) -> List[Document]:
        """
        Async version of rerank that doesn't block event loop.

        Device acceleration still works in thread pool.

        Args:
            documents: List of documents to rerank
            query: Query text
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, documents, query, top_k)
