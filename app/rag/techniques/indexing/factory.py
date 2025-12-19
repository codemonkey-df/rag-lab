"""
Factory for creating indexing strategy instances.

The IndexingStrategyFactory uses the registry pattern to map strategy names
to strategy classes and provides a centralized point for strategy creation
and configuration.
"""

import logging
from typing import Any, Dict

from app.rag.techniques.indexing.base import BaseIndexingStrategy
from app.rag.techniques.indexing.headers import HeadersStrategy
from app.rag.techniques.indexing.parent import ParentDocumentStrategy
from app.rag.techniques.indexing.proposition import PropositionStrategy
from app.rag.techniques.indexing.semantic import SemanticStrategy
from app.rag.techniques.indexing.standard import StandardStrategy

logger = logging.getLogger(__name__)


class IndexingStrategyFactory:
    """
    Factory for creating indexing strategy instances.

    Provides a centralized registry of available strategies and handles
    their instantiation with proper configuration validation.

    Usage:
        factory = IndexingStrategyFactory()
        strategy = factory.create("standard", chunk_size=1024, chunk_overlap=200)

    Extending:
        1. Create new strategy class implementing BaseIndexingStrategy
        2. Add to _STRATEGIES registry
        3. No changes needed to endpoint or pipeline
    """

    # Registry mapping strategy names to classes
    _STRATEGIES: Dict[str, type] = {
        "standard": StandardStrategy,
        "parent_document": ParentDocumentStrategy,
        "semantic": SemanticStrategy,
        "headers": HeadersStrategy,
        "proposition": PropositionStrategy,
    }

    @classmethod
    def get_available_strategies(cls) -> Dict[str, type]:
        """
        Get all available strategies.

        Returns:
            Dictionary mapping strategy names to their classes
        """
        return cls._STRATEGIES.copy()

    @classmethod
    def is_registered(cls, strategy_name: str) -> bool:
        """
        Check if a strategy is registered.

        Args:
            strategy_name: Name of the strategy

        Returns:
            True if strategy is registered, False otherwise
        """
        return strategy_name in cls._STRATEGIES

    @classmethod
    def create(cls, strategy_name: str, **config) -> BaseIndexingStrategy:
        """
        Create an indexing strategy instance.

        Args:
            strategy_name: Name of the strategy (standard, parent_document, etc.)
            **config: Strategy-specific configuration (passed through for validation)

        Returns:
            Instance of the requested strategy

        Raises:
            ValueError: If strategy is not registered
            ValueError: If config validation fails

        Example:
            strategy = factory.create("standard", chunk_size=1024, chunk_overlap=200)
        """
        if strategy_name not in cls._STRATEGIES:
            available = ", ".join(cls._STRATEGIES.keys())
            raise ValueError(
                f"Unknown strategy: {strategy_name}. Available strategies: {available}"
            )

        strategy_class = cls._STRATEGIES[strategy_name]
        strategy = strategy_class()

        # Validate and enrich config
        strategy.validate_config(config)
        enriched_config = strategy.enrich_config(config)

        logger.info(f"Created strategy: {strategy_name}")

        return strategy

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        """
        Register a new strategy (for extension/plugins).

        Args:
            name: Strategy name
            strategy_class: Class implementing BaseIndexingStrategy

        Raises:
            ValueError: If strategy_class doesn't implement BaseIndexingStrategy
            ValueError: If name already registered

        Example:
            class CustomStrategy(BaseIndexingStrategy):
                ...

            factory.register_strategy("custom", CustomStrategy)
        """
        # Validate that class implements BaseIndexingStrategy
        if not issubclass(strategy_class, BaseIndexingStrategy):
            raise ValueError(
                f"Strategy class must implement BaseIndexingStrategy, "
                f"got {strategy_class}"
            )

        if name in cls._STRATEGIES:
            raise ValueError(f"Strategy '{name}' is already registered")

        cls._STRATEGIES[name] = strategy_class
        logger.info(f"Registered new strategy: {name}")

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get information about a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy metadata

        Raises:
            ValueError: If strategy is not registered

        Example:
            info = factory.get_strategy_info("semantic")
            print(info["required_config"], info["async"], info["class_name"])
        """
        if not cls.is_registered(strategy_name):
            raise ValueError(f"Strategy not registered: {strategy_name}")

        strategy_class = cls._STRATEGIES[strategy_name]
        instance = strategy_class()

        return {
            "name": strategy_name,
            "class_name": strategy_class.__name__,
            "required_config": instance.get_required_config(),
            "optional_config": instance.get_optional_config(),
            "async_execution": instance.supports_async_execution(),
            "progress_stages": instance.get_progress_stages(),
        }

    @classmethod
    def list_strategies(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all strategies.

        Returns:
            Dictionary mapping strategy names to their info

        Example:
            strategies = factory.list_strategies()
            for name, info in strategies.items():
                print(f"{name}: async={info['async_execution']}")
        """
        return {name: cls.get_strategy_info(name) for name in cls._STRATEGIES.keys()}
