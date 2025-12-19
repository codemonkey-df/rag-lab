"""
Factory for creating filtering technique instances.

This factory provides a registry-based approach to instantiating filtering techniques
without coupling the pipeline to specific implementations.
"""

import logging

from app.models.enums import RAGTechnique
from app.rag.techniques.filtering.base import BaseFiltering
from app.rag.techniques.filtering.compression import CompressionFilter
from app.rag.techniques.filtering.reranking import RerankingFilter

logger = logging.getLogger(__name__)


class FilteringFactory:
    """
    Factory for creating filtering technique instances.

    Uses a registry pattern to map technique enums to implementation classes.
    Pre-populated with default filtering techniques.
    """

    # Pre-populated registry mapping technique enums to classes
    _REGISTRY = {
        RAGTechnique.RERANKING: RerankingFilter,
        RAGTechnique.CONTEXTUAL_COMPRESSION: CompressionFilter,
    }

    @classmethod
    def register(cls, technique: RAGTechnique, strategy_class: type):
        """
        Register a filtering strategy in the factory.

        Args:
            technique: RAGTechnique enum value
            strategy_class: Strategy class implementing BaseFiltering

        Raises:
            ValueError: If strategy_class doesn't implement BaseFiltering
        """
        if not issubclass(strategy_class, BaseFiltering):
            raise ValueError(f"{strategy_class.__name__} must implement BaseFiltering")
        cls._REGISTRY[technique] = strategy_class
        logger.info(
            f"Registered filtering strategy: {technique} -> {strategy_class.__name__}"
        )

    @classmethod
    def create(cls, technique: RAGTechnique, **kwargs) -> BaseFiltering:
        """
        Create a filtering instance.

        Args:
            technique: RAGTechnique enum value identifying the filtering technique
            **kwargs: Arguments to pass to the strategy constructor

        Returns:
            Instance of the specified filtering strategy

        Raises:
            ValueError: If technique is unknown or not registered
        """
        strategy_class = cls._REGISTRY.get(technique)
        if not strategy_class:
            raise ValueError(
                f"Unknown filtering technique: {technique}. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )
        return strategy_class(**kwargs)

    @classmethod
    def list_techniques(cls):
        """
        List all registered filtering techniques.

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
