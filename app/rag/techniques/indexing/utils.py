"""
Shared utilities for document ingestion.

Common functions used across multiple ingestion strategies.
"""

import asyncio
import logging
from typing import Any, Callable, List, Tuple, TypeVar

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

T = TypeVar("T")


def load_pdf_with_metadata(file_path: str) -> List[Document]:
    """
    Load PDF using LangChain PyMuPDFLoader with page/line metadata.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of Document objects with metadata:
        - page: page number
        - source: file path
        - (line numbers added during chunking)
    """
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Ensure metadata includes page numbers
    for i, doc in enumerate(documents):
        if "page" not in doc.metadata:
            doc.metadata["page"] = i + 1
        if "source" not in doc.metadata:
            doc.metadata["source"] = file_path

    logger.info(f"Loaded PDF: {len(documents)} pages from {file_path}")
    return documents


async def process_items_in_parallel(
    items: List[T],
    process_func: Callable[[T, int], Any],
    max_concurrent: int,
    item_name: str = "item",
) -> List[Tuple[Any, int, Exception | None]]:
    """
    Process a list of items in parallel with semaphore-based concurrency control.

    Args:
        items: List of items to process
        process_func: Async function that processes one item: async (item, index) -> result
        max_concurrent: Maximum number of concurrent operations
        item_name: Name for logging (e.g., "chunk", "proposition", "header")

    Returns:
        List of tuples: (result, index, error)
        - result: The return value from process_func (or None if error)
        - index: Original index of the item
        - error: Exception if processing failed, None if successful
    """
    if not items:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(item: T, index: int):
        async with semaphore:
            try:
                result = await process_func(item, index)
                return result, index, None
            except Exception as e:
                logger.warning(f"Failed to process {item_name} {index}: {e}")
                return None, index, e

    results = await asyncio.gather(
        *[process_with_semaphore(item, i) for i, item in enumerate(items)],
        return_exceptions=True,
    )

    # Handle exceptions from gather itself
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Unexpected error processing {item_name} {i}: {result}")
            processed_results.append((None, i, result))
        else:
            processed_results.append(result)

    return processed_results
