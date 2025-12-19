"""
Document ingestion pipeline and facade service.

Implements the Template Method pattern for consistent ingestion workflow
and the Facade pattern for a unified service interface.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from sqlmodel import Session

from app.db.repositories import DocumentRepository
from app.rag.techniques.indexing.base import BaseIndexingStrategy
from app.rag.techniques.indexing.factory import IndexingStrategyFactory
from app.rag.techniques.indexing.utils import load_pdf_with_metadata

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Template Method pattern implementation for document ingestion.

    Defines the common ingestion workflow skeleton:
    1. Load documents from file
    2. Chunk documents (strategy-specific)
    3. Post-process chunks (strategy-specific)
    4. Index chunks into vectorstore (strategy-specific)
    5. Update document status

    Each strategy implements the specific chunk(), post_process(), and index()
    methods while the pipeline handles common concerns like progress tracking,
    error handling, and status updates.
    """

    def __init__(
        self,
        strategy: BaseIndexingStrategy,
        document_id: UUID,
        db_session: Session,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            strategy: BaseIndexingStrategy instance to use
            document_id: UUID of the document being ingested
            db_session: Database session for status updates
        """
        self.strategy = strategy
        self.document_id = document_id
        self.db_session = db_session
        self.repo = DocumentRepository(db_session)

    async def _update_progress(self, progress: int, message: str = "") -> None:
        """
        Update document progress in database.

        Args:
            progress: Progress percentage (0-100)
            message: Optional log message
        """
        try:
            doc = self.repo.get_by_id(self.document_id)
            if doc:
                doc.indexing_progress = progress
                self.repo.update(doc)
                if message:
                    logger.debug(f"[{self.document_id}] {progress}%: {message}")
        except Exception as e:
            logger.warning(f"Failed to update progress: {e}")

    def _get_progress_callback(self, stage_start: int, stage_end: int) -> Callable:
        """
        Create a progress callback for a pipeline stage.

        Args:
            stage_start: Starting percentage for this stage
            stage_end: Ending percentage for this stage

        Returns:
            Callback function that updates progress within the stage range
        """

        async def callback(stage_progress: float) -> None:
            """Update progress within stage range."""
            overall_progress = int(
                stage_start + ((stage_progress / 100) * (stage_end - stage_start))
            )
            await self._update_progress(overall_progress)

        return callback

    async def ingest(
        self,
        file_path: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute the ingestion pipeline (Template Method).

        Args:
            file_path: Path to the document file
            config: Configuration for the strategy
            progress_callback: Optional external progress callback

        Returns:
            Dictionary with status and details

        Raises:
            Exception: If any stage fails (caught and handled by caller)
        """
        doc = self.repo.get_by_id(self.document_id)
        progress_stages = self.strategy.get_progress_stages()

        try:
            # Stage 1: Load documents
            await self._update_progress(
                progress_stages.get("loading", 10), "Loading PDF..."
            )
            documents = load_pdf_with_metadata(file_path)

            # Stage 2: Chunk documents
            chunk_progress = progress_stages.get("chunking", 30)
            await self._update_progress(chunk_progress - 5, "Chunking documents...")
            chunks = await self.strategy.chunk(documents, config)

            # Stage 3: Post-process chunks
            postprocess_progress = progress_stages.get("post_processing", 50)
            await self._update_progress(
                postprocess_progress - 10, "Post-processing chunks..."
            )

            # Pass progress callback to strategy if needed
            config_with_callback = {**config, "progress_callback": progress_callback}
            chunks = await self.strategy.post_process(chunks, config_with_callback)

            # Stage 4: Index chunks
            index_progress = progress_stages.get("indexing", 80)
            await self._update_progress(index_progress - 5, "Indexing chunks...")
            await self.strategy.index(self.document_id, chunks)

            # Stage 5: Complete
            complete_progress = progress_stages.get("complete", 100)
            doc.status = "completed"
            doc.indexing_progress = complete_progress
            doc.processed_at = datetime.utcnow()
            self.repo.update(doc)

            logger.info(
                f"Successfully ingested document {self.document_id} "
                f"with strategy {self.strategy.__class__.__name__}"
            )

            return {
                "status": "completed",
                "document_id": self.document_id,
                "chunks": len(chunks),
                "processed_at": doc.processed_at,
            }

        except Exception as e:
            logger.error(
                f"Error ingesting document {self.document_id}: {e}", exc_info=True
            )
            doc.status = "failed"
            self.repo.update(doc)
            raise


class DocumentIngestionService:
    """
    Facade pattern implementation for document ingestion.

    Provides a single, unified interface for the entire ingestion process.
    Hides the complexity of strategy selection, pipeline creation, and
    execution coordination.

    This is the primary entry point for the API endpoint. The endpoint
    should only call methods on this service, not create strategies or
    pipelines directly.

    Usage:
        service = DocumentIngestionService()
        result = await service.ingest_document(
            document_id=doc_id,
            file_path="/path/to/file.pdf",
            strategy_name="semantic",
            config={"chunk_size": 1024, "chunk_overlap": 200},
            db_session=db_session,
            background_tasks=background_tasks,
        )
    """

    def __init__(self):
        """Initialize the ingestion service."""
        self.factory = IndexingStrategyFactory()

    async def ingest_document(
        self,
        document_id: UUID,
        file_path: str,
        strategy_name: str,
        config: Dict[str, Any],
        db_session: Session,
        background_tasks: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document with the specified strategy.

        Handles strategy creation, pipeline execution, and background task
        scheduling based on the strategy's requirements.

        Args:
            document_id: UUID of the document to ingest
            file_path: Path to the document file
            strategy_name: Name of the indexing strategy to use
            config: Strategy configuration (chunk_size, chunk_overlap, etc.)
            db_session: Database session for status updates
            background_tasks: FastAPI BackgroundTasks for async execution (optional)

        Returns:
            Dictionary with status information:
            - status: "completed" or "processing"
            - document_id: UUID of the document
            - chunks: Number of chunks created (if completed)
            - processed_at: Datetime when ingestion completed (if completed)

        Raises:
            ValueError: If strategy is not registered or config is invalid
            Exception: Various exceptions from strategy or pipeline execution
        """
        logger.info(
            f"Starting ingestion: doc_id={document_id}, strategy={strategy_name}"
        )

        # Create strategy from factory
        try:
            strategy = self.factory.create(strategy_name, **config)
        except ValueError as e:
            logger.error(f"Failed to create strategy: {e}")
            raise

        # Create pipeline
        pipeline = IngestionPipeline(strategy, document_id, db_session)

        # Check if strategy should run async
        if strategy.supports_async_execution() and background_tasks:
            # Run in background
            background_tasks.add_task(
                self._run_ingestion_async,
                pipeline,
                file_path,
                config,
            )
            return {
                "status": "processing",
                "document_id": document_id,
                "message": f"Document queued for ingestion with {strategy_name} strategy",
            }
        else:
            # Run synchronously with lock for immediate strategies
            from app.core.concurrency import llm_lock_manager

            await llm_lock_manager.acquire_for_indexing()
            try:
                result = await pipeline.ingest(file_path, config)
                return {
                    "status": result["status"],
                    "document_id": document_id,
                    "chunks": result.get("chunks"),
                    "processed_at": result.get("processed_at"),
                }
            finally:
                await llm_lock_manager.release_for_indexing()

    async def _run_ingestion_async(
        self,
        pipeline: IngestionPipeline,
        file_path: str,
        config: Dict[str, Any],
    ) -> None:
        """
        Run ingestion pipeline asynchronously (for background tasks).

        Args:
            pipeline: IngestionPipeline instance
            file_path: Path to the document file
            config: Strategy configuration

        Note:
            Exceptions are caught and logged. Status is updated to failed.
        """
        try:
            await pipeline.ingest(file_path, config)
        except Exception as e:
            logger.error(f"Async ingestion failed: {e}", exc_info=True)

    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available strategies.

        Returns:
            Dictionary mapping strategy names to their metadata

        Example:
            strategies = service.get_available_strategies()
            for name, info in strategies.items():
                print(f"{name}: async={info['async_execution']}")
        """
        return self.factory.list_strategies()

    def validate_strategy_config(
        self, strategy_name: str, config: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate configuration for a strategy.

        Args:
            strategy_name: Name of the strategy
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if config is valid
            - error_message: None if valid, error description if invalid
        """
        try:
            strategy = self.factory.create(strategy_name, **config)
            return True, None
        except ValueError as e:
            return False, str(e)
