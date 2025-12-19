"""
Base interface for query expansion techniques.

Query expansion techniques enhance the user's query to improve retrieval results.
Examples: HyDE (Hypothetical Document Embedding), query reformulation, etc.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseQueryExpansion(ABC):
    """
    Abstract base class for all query expansion techniques.

    Query expansion techniques take a user query and return an expanded/enhanced version
    that may include multiple queries, hypothetical documents, or reformulated questions.
    """

    @abstractmethod
    async def expand(self, query: str) -> str:
        """
        Expand or enhance the query.

        Args:
            query: Original user query

        Returns:
            Expanded query string (can contain multiple queries, hypothetical documents, etc.)

        Raises:
            Exception: If query expansion fails
        """
        pass

    def get_name(self) -> str:
        """
        Get the name of this expansion technique.

        Returns:
            Human-readable name
        """
        return self.__class__.__name__
