"""
Factory for creating query expansion technique instances.

This factory provides a registry-based approach to instantiating query expansion techniques
without coupling the pipeline to specific implementations.
"""

import logging

from app.models.enums import RAGTechnique
from app.rag.techniques.query_expansion.base import BaseQueryExpansion
from app.rag.techniques.query_expansion.hyde import HyDEExpansion

logger = logging.getLogger(__name__)


class QueryExpansionFactory:
    """
    Factory for creating query expansion technique instances.

    Uses a registry pattern to map technique enums to implementation classes.
    Pre-populated with default query expansion techniques.
    """

    # Pre-populated registry mapping technique enums to classes
    _REGISTRY = {
        RAGTechnique.HYDE: HyDEExpansion,
    }

    @classmethod
    def register(cls, technique: RAGTechnique, strategy_class: type):
        """
        Register a query expansion strategy in the factory.

        Args:
            technique: RAGTechnique enum value
            strategy_class: Strategy class implementing BaseQueryExpansion

        Raises:
            ValueError: If strategy_class doesn't implement BaseQueryExpansion
        """
        if not issubclass(strategy_class, BaseQueryExpansion):
            raise ValueError(
                f"{strategy_class.__name__} must implement BaseQueryExpansion"
            )
        cls._REGISTRY[technique] = strategy_class
        logger.info(
            f"Registered query expansion strategy: {technique} -> {strategy_class.__name__}"
        )

    @classmethod
    def create(cls, technique: RAGTechnique, **kwargs) -> BaseQueryExpansion:
        """
        Create a query expansion instance.

        Args:
            technique: RAGTechnique enum value identifying the expansion technique
            **kwargs: Arguments to pass to the strategy constructor

        Returns:
            Instance of the specified query expansion strategy

        Raises:
            ValueError: If technique is unknown or not registered
        """
        strategy_class = cls._REGISTRY.get(technique)
        if not strategy_class:
            raise ValueError(
                f"Unknown query expansion technique: {technique}. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )
        return strategy_class(**kwargs)

    @classmethod
    def list_techniques(cls):
        """
        List all registered query expansion techniques.

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
