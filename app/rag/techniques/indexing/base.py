"""
Base interface for all indexing strategies.

This module provides the abstract base class that all indexing strategies must implement.
It defines the common interface and lifecycle for document chunking and indexing.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from uuid import UUID

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseIndexingStrategy(ABC):
    """
    Abstract base class for all indexing strategies.

    Each strategy encapsulates a specific approach to chunking and indexing documents.
    Strategies are responsible for:
    - Loading documents (if strategy-specific)
    - Chunking documents according to strategy parameters
    - Post-processing chunks (e.g., adding headers, generating propositions)
    - Indexing chunks into the vectorstore

    The IngestionPipeline uses this interface to execute strategies consistently
    while allowing each strategy to implement custom logic where needed.
    """

    def __init__(self):
        """Initialize the strategy."""
        pass

    @abstractmethod
    async def chunk(
        self,
        documents: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Chunk documents according to this strategy.

        Args:
            documents: List of Document objects to chunk
            config: Configuration dictionary with strategy-specific parameters
                   Common keys: chunk_size, chunk_overlap
                   Strategy-specific: base_chunking, breakpoint_threshold_*, etc.

        Returns:
            List of chunked Document objects with metadata

        Raises:
            ValueError: If required config keys are missing
        """
        pass

    @abstractmethod
    async def post_process(
        self,
        chunks: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Post-process chunks (e.g., add headers, generate propositions).

        Some strategies don't need post-processing. In that case,
        simply return chunks unchanged.

        Args:
            chunks: List of chunked Document objects
            config: Configuration dictionary with strategy-specific parameters

        Returns:
            List of post-processed Document objects

        Raises:
            Exception: Strategy-specific exceptions during processing
        """
        pass

    @abstractmethod
    async def index(
        self,
        document_id: UUID,
        chunks: List[Document],
    ) -> None:
        """
        Index chunks into the vectorstore.

        Args:
            document_id: UUID of the document being indexed
            chunks: List of Document objects to index

        Raises:
            Exception: If vectorstore operation fails
        """
        pass

    def get_required_config(self) -> List[str]:
        """
        Return list of required config keys for this strategy.

        Override this method if your strategy requires specific config keys.
        The IngestionPipeline uses this to validate config before execution.

        Returns:
            List of required configuration keys

        Example:
            ["chunk_size", "chunk_overlap", "base_chunking"]
        """
        return ["chunk_size", "chunk_overlap"]

    def get_optional_config(self) -> Dict[str, Any]:
        """
        Return optional config with defaults for this strategy.

        Override this method to provide default values for optional parameters.

        Returns:
            Dictionary mapping config keys to default values
        """
        return {}

    def supports_async_execution(self) -> bool:
        """
        Whether this strategy should run in a background task.

        Fast strategies (standard, parent) typically return False
        and execute immediately. Slow strategies (semantic, headers,
        proposition) return True to avoid blocking the endpoint.

        Returns:
            True if strategy should run as background task, False otherwise
        """
        return False

    def get_progress_stages(self) -> Dict[str, int]:
        """
        Return progress checkpoint percentages for this strategy.

        Override to customize progress stages. Used to track ingestion progress
        in the UI.

        Returns:
            Dict mapping stage name to percentage (e.g., {'loading': 10, 'chunking': 30})
        """
        return {
            "loading": 10,
            "chunking": 30,
            "post_processing": 50,
            "indexing": 80,
            "complete": 100,
        }

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate that config contains all required keys.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If required config keys are missing or invalid
        """
        required = self.get_required_config()
        missing = [key for key in required if key not in config]

        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires config keys: {missing}. "
                f"Got: {list(config.keys())}"
            )

    def enrich_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich config with optional defaults and strategy-specific values.

        Args:
            config: Configuration dictionary to enrich

        Returns:
            Enhanced configuration dictionary
        """
        enriched = {**config}
        defaults = self.get_optional_config()
        for key, default_value in defaults.items():
            if key not in enriched:
                enriched[key] = default_value
        return enriched
