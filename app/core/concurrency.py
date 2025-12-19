"""
Resource lock management to prevent concurrent Ollama calls
"""
import asyncio
import logging

logger = logging.getLogger(__name__)


class LlmLockManager:
    """
    Manages concurrent access to Ollama LLM to prevent resource exhaustion.
    
    Uses a semaphore to allow parallel operations within a single query
    (e.g., HyDE + Retrieval) while still controlling overall concurrency.
    
    Rules:
    - Indexing (background) acquires all semaphore slots and sets _indexing_active = True
    - Querying (foreground) checks _indexing_active first
    - If indexing is active, query returns False (should return 503)
    - Multiple operations within same query can run in parallel (up to max_concurrent)
    - Prevents resource exhaustion while enabling parallel HyDE + Retrieval
    """
    
    def __init__(self, max_concurrent: int = 4):
        """
        Initialize LlmLockManager with semaphore for parallel operations.
        
        Args:
            max_concurrent: Maximum concurrent Ollama operations (default: 4)
                           Should match OLLAMA_NUM_PARALLEL setting
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._indexing_lock = asyncio.Lock()
        self._indexing_active = False
        self._max_concurrent = max_concurrent
        logger.info(f"LlmLockManager initialized with max_concurrent={max_concurrent}")
    
    async def acquire_for_indexing(self):
        """
        Acquire all semaphore slots for indexing operation.
        
        This prevents any queries from running during indexing.
        """
        await self._indexing_lock.acquire()
        self._indexing_active = True
        # Acquire all semaphore slots to block queries
        for _ in range(self._max_concurrent):
            await self._semaphore.acquire()
        logger.info("Lock acquired for indexing (all semaphore slots)")
    
    async def release_for_indexing(self):
        """Release all semaphore slots after indexing completes"""
        # Release all semaphore slots
        for _ in range(self._max_concurrent):
            self._semaphore.release()
        self._indexing_active = False
        self._indexing_lock.release()
        logger.info("Lock released after indexing (all semaphore slots)")
    
    async def acquire_for_query(self) -> bool:
        """
        Attempt to acquire semaphore slot for query operation.
        
        This allows multiple operations within the same query to run in parallel
        (e.g., HyDE + Retrieval) while still controlling overall concurrency.
        
        Returns:
            True if slot acquired, False if indexing is active
        """
        if self._indexing_active:
            logger.warning("Query rejected: indexing in progress")
            return False
        
        # Acquire one semaphore slot (allows parallel ops within query)
        await self._semaphore.acquire()
        logger.debug("Semaphore slot acquired for query")
        return True
    
    async def release_for_query(self):
        """Release semaphore slot after query completes"""
        self._semaphore.release()
        logger.debug("Semaphore slot released after query")
    
    @property
    def is_indexing_active(self) -> bool:
        """Check if indexing is currently active"""
        return self._indexing_active


# Global singleton instance
llm_lock_manager = LlmLockManager()
