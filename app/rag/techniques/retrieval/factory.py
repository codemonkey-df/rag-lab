"""
Factory for creating retrieval technique instances.

This factory provides a registry-based approach to instantiating retrieval techniques
without coupling the pipeline to specific implementations.
"""

import logging

from app.models.enums import RAGTechnique
from app.rag.techniques.retrieval.base import BaseRetrieval
from app.rag.techniques.retrieval.basic import BasicRetrieval
from app.rag.techniques.retrieval.hybrid_retrieval import HybridRetrieval

logger = logging.getLogger(__name__)


class RetrievalFactory:
    """
    Factory for creating retrieval technique instances.

    Uses a registry pattern to map technique enums to implementation classes.
    Pre-populated with default retrieval techniques.
    """

    # Pre-populated registry mapping technique enums to classes
    _REGISTRY = {
        RAGTechnique.BASIC_RAG: BasicRetrieval,
        RAGTechnique.FUSION_RETRIEVAL: HybridRetrieval,
    }

    @classmethod
    def register(cls, technique: RAGTechnique, strategy_class: type):
        """
        Register a retrieval strategy in the factory.

        Args:
            technique: RAGTechnique enum value
            strategy_class: Strategy class implementing BaseRetrieval

        Raises:
            ValueError: If strategy_class doesn't implement BaseRetrieval
        """
        if not issubclass(strategy_class, BaseRetrieval):
            raise ValueError(f"{strategy_class.__name__} must implement BaseRetrieval")
        cls._REGISTRY[technique] = strategy_class
        logger.info(
            f"Registered retrieval strategy: {technique} -> {strategy_class.__name__}"
        )

    @classmethod
    def create(cls, technique: RAGTechnique, **kwargs) -> BaseRetrieval:
        """
        Create a retrieval instance.

        Args:
            technique: RAGTechnique enum value identifying the retrieval technique
            **kwargs: Arguments to pass to the strategy constructor

        Returns:
            Instance of the specified retrieval strategy

        Raises:
            ValueError: If technique is unknown or not registered
        """
        strategy_class = cls._REGISTRY.get(technique)
        if not strategy_class:
            raise ValueError(
                f"Unknown retrieval technique: {technique}. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )
        return strategy_class(**kwargs)

    @classmethod
    def list_techniques(cls):
        """
        List all registered retrieval techniques.

        Returns:
            List of registered RAGTechnique values
        """
        return list(cls._REGISTRY.keys())

    @classmethod
    def is_registered(cls, technique: RAGTechnique) -> bool:
        """
        Check if a technique is registered.

        Args:
            technique: RAGTechnique enum value

        Returns:
            True if registered, False otherwise
        """
        return technique in cls._REGISTRY
