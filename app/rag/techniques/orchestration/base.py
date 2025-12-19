"""
Base interface for orchestration techniques.

Orchestration techniques manage their own complete RAG pipeline with special logic.
Examples: Self-RAG, CRAG (Corrective RAG), Adaptive Retrieval, etc.

These techniques are "advanced controllers" that take over the entire pipeline,
making decisions about retrieval, evaluation, and generation themselves.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseOrchestration(ABC):
    """
    Abstract base class for all orchestration techniques.

    Orchestration techniques implement their own complete RAG pipeline.
    They control how retrieval, filtering, and generation are performed,
    often with sophisticated decision-making and quality evaluation.

    Unlike standard techniques (retrieval, filtering, etc.), orchestration
    techniques are mutually exclusive and should not be combined.
    """

    @abstractmethod
    async def process(
        self,
        query: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute the orchestration technique's complete pipeline.

        Args:
            query: User query
            **kwargs: Additional parameters (top_k, temperature, user_context, etc.)

        Returns:
            Dictionary containing:
                - answer (str): Generated answer
                - documents (List[Document]): Retrieved documents used
                - metadata (Dict): Technique-specific metadata
                - Additional fields as needed by specific techniques

        Raises:
            Exception: If processing fails
        """
        pass

    def get_name(self) -> str:
        """
        Get the name of this orchestration technique.

        Returns:
            Human-readable name
        """
        return self.__class__.__name__
