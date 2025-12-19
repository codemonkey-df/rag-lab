"""
Factory for creating orchestration technique instances.

This factory provides a registry-based approach to instantiating orchestration techniques
without coupling the pipeline to specific implementations.
"""

import logging

from app.models.enums import RAGTechnique
from app.rag.techniques.orchestration.adaptive import AdaptiveRetrieval
from app.rag.techniques.orchestration.base import BaseOrchestration
from app.rag.techniques.orchestration.crag import CRAG
from app.rag.techniques.orchestration.self_rag import SelfRAG

logger = logging.getLogger(__name__)


class OrchestrationFactory:
    """
    Factory for creating orchestration technique instances.

    Uses a registry pattern to map technique enums to implementation classes.
    Pre-populated with default orchestration techniques.
    """

    # Pre-populated registry mapping technique enums to classes
    _REGISTRY = {
        RAGTechnique.SELF_RAG: SelfRAG,
        RAGTechnique.CRAG: CRAG,
        RAGTechnique.ADAPTIVE_RETRIEVAL: AdaptiveRetrieval,
    }

    @classmethod
    def register(cls, technique: RAGTechnique, strategy_class: type):
        """
        Register an orchestration strategy in the factory.

        Args:
            technique: RAGTechnique enum value
            strategy_class: Strategy class implementing BaseOrchestration

        Raises:
            ValueError: If strategy_class doesn't implement BaseOrchestration
        """
        if not issubclass(strategy_class, BaseOrchestration):
            raise ValueError(
                f"{strategy_class.__name__} must implement BaseOrchestration"
            )
        cls._REGISTRY[technique] = strategy_class
        logger.info(
            f"Registered orchestration strategy: {technique} -> {strategy_class.__name__}"
        )

    @classmethod
    def create(cls, technique: RAGTechnique, **kwargs) -> BaseOrchestration:
        """
        Create an orchestration instance.

        Args:
            technique: RAGTechnique enum value identifying the orchestration technique
            **kwargs: Arguments to pass to the strategy constructor

        Returns:
            Instance of the specified orchestration strategy

        Raises:
            ValueError: If technique is unknown or not registered
        """
        strategy_class = cls._REGISTRY.get(technique)
        if not strategy_class:
            raise ValueError(
                f"Unknown orchestration technique: {technique}. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )
        return strategy_class(**kwargs)

    @classmethod
    def list_techniques(cls):
        """
        List all registered orchestration techniques.

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
