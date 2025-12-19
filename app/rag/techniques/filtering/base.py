"""
Base interface for filtering techniques.

Filtering techniques filter, rerank, or compress retrieved documents.
Examples: Reranking, contextual compression, etc.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseFiltering(ABC):
    """
    Abstract base class for all filtering techniques.

    Filtering techniques process retrieved documents to improve quality by:
    - Reranking: Re-ordering by relevance
    - Compression: Extracting relevant portions
    - Deduplication: Removing duplicates
    - Other document refinement operations
    """

    @abstractmethod
    async def filter(
        self,
        documents: List[Document],
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Document]:
        """
        Filter, rerank, or compress documents.

        Args:
            documents: List of retrieved documents
            query: Original user query (for reranking context)
            top_k: Target number of documents after filtering
            **kwargs: Additional parameters specific to filtering technique

        Returns:
            Filtered/reranked list of Document objects

        Raises:
            Exception: If filtering fails
        """
        pass

    def get_name(self) -> str:
        """
        Get the name of this filtering technique.

        Returns:
            Human-readable name
        """
        return self.__class__.__name__
