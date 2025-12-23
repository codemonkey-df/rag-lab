"""
Pytest configuration and fixtures for indexing strategy tests.

Provides common fixtures for API client, test documents, and utilities.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from uuid import UUID

import httpx
import pytest

logger = logging.getLogger(__name__)

# Try to import vectorstore utilities for chunk counting
try:
    from app.services.vectorstore import get_chroma_collection

    VECTORSTORE_AVAILABLE = True
except ImportError:
    VECTORSTORE_AVAILABLE = False


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="Base URL of the API server to test against",
    )
    parser.addoption(
        "--test-pdf",
        action="store",
        default="tests/test.pdf",
        help="Path to test PDF file (relative to project root)",
    )
    parser.addoption(
        "--max-wait",
        action="store",
        type=int,
        default=600,
        help="Maximum wait time for document processing in seconds",
    )
    parser.addoption(
        "--timeout",
        action="store",
        type=int,
        default=300,
        help="Request timeout in seconds",
    )


@pytest.fixture(scope="session")
def server_url(pytestconfig) -> str:
    """Get server URL from command-line option or default."""
    return pytestconfig.getoption("--server-url").rstrip("/")


@pytest.fixture(scope="session")
def test_pdf_path(pytestconfig) -> Path:
    """Get path to test PDF file."""
    pdf_name = pytestconfig.getoption("--test-pdf")
    pdf_path = Path(pdf_name)
    if not pdf_path.is_absolute():
        # First try relative to project root
        project_root = Path(__file__).parent.parent
        pdf_path = project_root / pdf_name
        # If not found, try relative to tests folder
        if not pdf_path.exists():
            tests_dir = Path(__file__).parent
            pdf_path = tests_dir / pdf_name
    if not pdf_path.exists():
        raise FileNotFoundError(f"Test PDF not found: {pdf_path}")
    return pdf_path


@pytest.fixture(scope="session")
def max_wait_time(pytestconfig) -> int:
    """Get maximum wait time from command-line option."""
    return pytestconfig.getoption("--max-wait")


@pytest.fixture(scope="session")
def request_timeout(pytestconfig) -> int:
    """Get request timeout from command-line option."""
    return pytestconfig.getoption("--timeout")


@pytest.fixture(scope="session")
def api_client(server_url: str, request_timeout: int) -> httpx.Client:
    """
    Create httpx client for API requests.

    Args:
        server_url: Base URL of the API server
        request_timeout: Request timeout in seconds

    Yields:
        httpx.Client instance
    """
    client = httpx.Client(
        base_url=server_url,
        timeout=request_timeout,
        follow_redirects=True,
    )
    yield client
    client.close()


@pytest.fixture(scope="session")
def server_health_check(api_client: httpx.Client) -> bool:
    """
    Check if server is healthy and accessible.

    Raises:
        pytest.skip: If server is not accessible
    """
    try:
        response = api_client.get("/health", timeout=5.0)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("healthy", False):
                return True
        pytest.skip(f"Server not healthy at {api_client.base_url}")
    except Exception as e:
        pytest.skip(f"Cannot connect to server at {api_client.base_url}: {e}")


@pytest.fixture
def vectorstore_available() -> bool:
    """Check if vectorstore utilities are available."""
    return VECTORSTORE_AVAILABLE


@pytest.fixture(scope="session")
def llm_available() -> bool:
    """
    Check if LLM is available for headers/proposition strategies.

    Returns:
        True if LLM is available, False otherwise
    """
    try:
        import asyncio

        from app.core.health import check_ollama_health

        # Run async health check
        # Use asyncio.run() which is the recommended approach in Python 3.7+
        # It creates a new event loop, runs the coroutine, and closes the loop
        try:
            is_healthy, message = asyncio.run(check_ollama_health())
            if not is_healthy:
                logger.warning(f"LLM not available: {message}")
            return is_healthy
        except RuntimeError:
            # If there's already a running event loop (shouldn't happen in pytest fixture)
            # fall back to creating a new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                is_healthy, message = loop.run_until_complete(check_ollama_health())
                if not is_healthy:
                    logger.warning(f"LLM not available: {message}")
                return is_healthy
            finally:
                loop.close()
    except Exception as e:
        logger.warning(f"Could not check LLM availability: {e}")
        return False


@pytest.fixture
def require_llm(llm_available: bool):
    """
    Fixture that skips tests if LLM is not available.

    Use this for headers and proposition strategy tests.
    """
    if not llm_available:
        pytest.skip(
            "LLM not available - required for this test. Ensure Ollama is running."
        )
    return llm_available


def wait_for_processing(
    api_client: httpx.Client,
    document_id: UUID,
    max_wait: int = 600,
    poll_interval: int = 2,
) -> Dict[str, Any]:
    """
    Wait for document processing to complete.

    Args:
        api_client: httpx client for API requests
        document_id: UUID of the document
        max_wait: Maximum time to wait in seconds
        poll_interval: Time between status checks in seconds

    Returns:
        Document status dictionary

    Raises:
        TimeoutError: If processing times out
        Exception: If processing fails
    """
    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = api_client.get(f"/api/v1/documents/{document_id}")
        if response.status_code == 200:
            doc = response.json()
            status = doc.get("status")
            if status == "completed":
                return doc
            elif status == "failed":
                error_msg = doc.get("error", "Unknown error")
                raise Exception(f"Document processing failed: {error_msg}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Document processing timed out after {max_wait}s")


def upload_document(
    api_client: httpx.Client,
    file_path: Path,
    strategy: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    base_chunking: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload document to server with specific strategy and config.

    Args:
        api_client: httpx client for API requests
        file_path: Path to PDF file
        strategy: Indexing strategy name
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        base_chunking: Base chunking strategy (for headers)

    Returns:
        Dict with document_id and status
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/pdf")}
        data = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunking_strategy": strategy,
        }

        # Add strategy-specific parameters
        if base_chunking:
            data["base_chunking"] = base_chunking

        response = api_client.post("/api/v1/documents/upload", files=files, data=data)
        response.raise_for_status()
        return response.json()


def get_document_details(api_client: httpx.Client, document_id: UUID) -> Dict[str, Any]:
    """
    Get document details from API.

    Args:
        api_client: httpx client for API requests
        document_id: UUID of the document

    Returns:
        Document details dictionary
    """
    response = api_client.get(f"/api/v1/documents/{document_id}")
    response.raise_for_status()
    return response.json()


def get_chunks_from_vectorstore(
    document_id: UUID, sample_size: int = 5
) -> Dict[str, Any]:
    """
    Retrieve chunks from ChromaDB vectorstore with full text content.

    Args:
        document_id: UUID of the document
        sample_size: Number of chunks to retrieve

    Returns:
        Dictionary with chunk data:
        - chunk_count: Total number of chunks
        - chunk_samples: List of sample chunks with full_text field
    """
    if not VECTORSTORE_AVAILABLE:
        return {"chunk_count": None, "chunk_samples": []}

    try:
        collection = get_chroma_collection(document_id)
        # Get all chunks using get() method
        try:
            all_data = collection.get()
            if all_data and all_data.get("ids"):
                chunk_count = len(all_data["ids"])
                # Sample chunks using utility function
                from tests.utils.verification import get_chunk_samples

                chunk_samples = get_chunk_samples(all_data, sample_size=sample_size)
                return {"chunk_count": chunk_count, "chunk_samples": chunk_samples}
            else:
                return {"chunk_count": 0, "chunk_samples": []}
        except Exception:
            # Fallback: try similarity_search
            try:
                results = collection.similarity_search("", k=10000)
                chunk_count = len(results) if results else 0
                if results:
                    chunk_samples = [
                        {
                            "text": r.page_content[:200],  # First 200 chars for display
                            "full_text": r.page_content,  # Full text for verification
                            "text_length": len(r.page_content),
                            "metadata": r.metadata,
                        }
                        for r in results[:sample_size]
                    ]
                else:
                    chunk_samples = []
                return {"chunk_count": chunk_count, "chunk_samples": chunk_samples}
            except Exception:
                return {"chunk_count": None, "chunk_samples": []}
    except Exception:
        return {"chunk_count": None, "chunk_samples": []}


def delete_document(api_client: httpx.Client, document_id: UUID) -> None:
    """
    Delete document from server.

    Args:
        api_client: httpx client for API requests
        document_id: UUID of the document
    """
    try:
        api_client.delete(f"/api/v1/documents/{document_id}")
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def cleanup_document(api_client: httpx.Client) -> Generator:
    """
    Fixture to track and cleanup documents after tests.

    Yields:
        Function to register document for cleanup
    """
    document_ids = []

    def register(document_id: UUID):
        document_ids.append(document_id)

    yield register

    # Cleanup all registered documents
    for doc_id in document_ids:
        delete_document(api_client, doc_id)


def perform_retrieval_test(
    api_client: httpx.Client,
    document_id: UUID,
    query: str = "What is the main topic?",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Perform retrieval test with a query to verify indexing worked.

    Args:
        api_client: httpx client for API requests
        document_id: UUID of the document
        query: Query string
        top_k: Number of chunks to retrieve

    Returns:
        Retrieval result dictionary
    """
    payload = {
        "document_id": str(document_id),
        "query": query,
        "techniques": ["basic_rag"],
        "query_params": {"top_k": top_k},
    }

    try:
        start_time = time.time()
        response = api_client.post("/api/v1/rag/query", json=payload)
        elapsed = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "retrieved_chunks": result.get("retrieved_chunks", []),
                "chunks_count": len(result.get("retrieved_chunks", [])),
                "latency_ms": elapsed,
            }
        else:
            error_data = response.json() if response.content else {}
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "detail": error_data.get("detail", "Unknown error"),
                "latency_ms": elapsed,
            }
    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "Timeout",
            "detail": "Request timed out",
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Request Error",
            "detail": str(e),
        }
