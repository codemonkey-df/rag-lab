"""
Base interface for retrieval techniques.

Retrieval techniques fetch relevant documents from the vectorstore.
Examples: Vector retrieval, BM25 retrieval, hybrid retrieval, etc.
"""

import logging
from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseRetrieval(ABC):
    """
    Abstract base class for all retrieval techniques.

    Retrieval techniques are responsible for fetching relevant documents
    from the vectorstore based on a query.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        document_id: UUID,
        top_k: int = 5,
        **kwargs,
    ) -> List[Document]:
        """
        Retrieve relevant documents for the query.

        Args:
            query: User query or expanded query
            document_id: UUID of the document to query
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters (bm25_weight, chunking_strategy, etc.)

        Returns:
            List of retrieved Document objects

        Raises:
            Exception: If retrieval fails
        """
        pass

    def get_name(self) -> str:
        """
        Get the name of this retrieval technique.

        Returns:
            Human-readable name
        """
        return self.__class__.__name__
