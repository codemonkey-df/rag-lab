#!/usr/bin/env python3
"""
RAG Technique Combination Test Suite

Tests all valid combinations of Layer 2 and Layer 3 RAG techniques
against a FastAPI server.

With performance optimizations enabled:
- Pipeline Reordering (Funnel Strategy): Retrieve → Rerank → Compress (75% compression time reduction)
- HyDE uses hypothetical document for retrieval (correct implementation)
- Semaphore-based concurrency (allows parallel operations within query)
- Smaller model for HyDE (llama3.2:3b) with optimized parameters
- Parallel compression (only top 5 documents after reranking)
- Async reranking with GPU/MPS support
- Increased retrieval size (25+ docs) when reranking enabled

Usage:
    # Using uv (recommended)
    uv run test_rag_combinations.py -d test.pdf -q "Your query here"

    # With custom timeout for performance testing
    uv run test_rag_combinations.py -d test.pdf -q "Your query" --timeout 120

    # Or with python (requires httpx to be installed)
    python test_rag_combinations.py -d test.pdf -q "Your query here"

Requirements:
    - httpx (included in project dependencies)
    - FastAPI server running on localhost:8000 (or specify with --server)
    - Ollama with models: llama3.2:3b (for HyDE), main LLM model, nomic-embed-text
    - Ollama configured with OLLAMA_NUM_PARALLEL=4 and OLLAMA_MAX_LOADED_MODELS=2 (CRITICAL for performance)

Expected Performance (with optimizations):
    - basic_rag + hyde: ~3-4s (down from 22.9s)
    - basic_rag + hyde + reranking: ~6-8s (down from 16.8s)
    - basic_rag + hyde + reranking + compression: ~10-12s (down from 26.9s)
    - fusion + hyde + reranking + compression: ~12-14s (down from 27s)
    - Most combinations: 2-14s (previously 15-30s)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx

# Technique definitions
LAYER_2_BASE = ["basic_rag", "fusion_retrieval"]
LAYER_2_ADDONS = ["hyde", "reranking", "contextual_compression"]
LAYER_3 = ["self_rag", "crag", "adaptive_retrieval"]


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test all valid RAG technique combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-d", "--document", required=True, help="Path to PDF file")
    parser.add_argument("-q", "--query", required=True, help="Query string to test")
    parser.add_argument(
        "-s",
        "--server",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--session-id", type=str, help="Optional session ID (for existing document)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)"
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.5,
        help="BM25 weight for fusion (default: 0.5)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="LLM temperature (default: 0.7)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60, should be sufficient with optimizations)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a warmup query before testing (helps with model loading)",
    )
    return parser.parse_args()


def generate_valid_combinations() -> List[List[str]]:
    """
    Generate all valid Layer 2 + Layer 3 technique combinations.

    Returns:
        List of technique combination lists
    """
    combinations_list = []

    # Layer 2 combinations with basic_rag
    base_rag_combos = [
        ["basic_rag"],
        ["basic_rag", "hyde"],
        ["basic_rag", "reranking"],
        ["basic_rag", "contextual_compression"],
        ["basic_rag", "hyde", "reranking"],
        ["basic_rag", "hyde", "contextual_compression"],
        ["basic_rag", "reranking", "contextual_compression"],
        ["basic_rag", "hyde", "reranking", "contextual_compression"],
    ]
    combinations_list.extend(base_rag_combos)

    # Layer 2 combinations with fusion_retrieval
    fusion_combos = [
        ["fusion_retrieval"],
        ["fusion_retrieval", "hyde"],
        ["fusion_retrieval", "reranking"],
        ["fusion_retrieval", "contextual_compression"],
        ["fusion_retrieval", "hyde", "reranking"],
        ["fusion_retrieval", "hyde", "contextual_compression"],
        ["fusion_retrieval", "reranking", "contextual_compression"],
        ["fusion_retrieval", "hyde", "reranking", "contextual_compression"],
    ]
    combinations_list.extend(fusion_combos)

    # Layer 3 combinations (standalone or with fusion)
    layer3_combos = [
        ["self_rag"],
        ["self_rag", "fusion_retrieval"],
        ["crag"],
        ["crag", "fusion_retrieval"],
        ["adaptive_retrieval"],
    ]
    combinations_list.extend(layer3_combos)

    return combinations_list


class RAGTester:
    """Test suite for RAG technique combinations."""

    def __init__(self, server_url: str, timeout: int = 60):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def check_server(self) -> bool:
        """Check if server is accessible."""
        try:
            response = self.client.get(f"{self.server_url}/health", timeout=5.0)
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get("healthy", False)
            return False
        except Exception:
            return False

    def upload_document(
        self, file_path: Path, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload document to server.

        Returns:
            Dict with document_id and session_id
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/pdf")}
            data = {
                "chunk_size": 1024,
                "chunk_overlap": 200,
                "chunking_strategy": "standard",
            }
            if session_id:
                data["session_id"] = session_id

            response = self.client.post(
                f"{self.server_url}/api/v1/documents/upload", files=files, data=data
            )
            response.raise_for_status()
            result = response.json()

            # Wait for processing if needed
            if result.get("status") == "processing":
                document_id = result["document_id"]
                self._wait_for_processing(document_id)

            return result

    def _wait_for_processing(self, document_id: UUID, max_wait: int = 300):
        """
        Wait for document processing to complete.

        Note: Queries against documents not in "completed" status will fail with HTTP 400
        and the error "Document is not ready for querying".
        """
        start_time = time.time()
        while time.time() - start_time < max_wait:
            response = self.client.get(
                f"{self.server_url}/api/v1/documents/{document_id}"
            )
            if response.status_code == 200:
                doc = response.json()
                if doc.get("status") == "completed":
                    return
                elif doc.get("status") == "failed":
                    raise Exception(
                        f"Document processing failed: {doc.get('error', 'Unknown error')}"
                    )
            time.sleep(2)
        raise TimeoutError(f"Document processing timed out after {max_wait}s")

    def execute_query(
        self,
        document_id: UUID,
        query: str,
        techniques: List[str],
        query_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute RAG query with given techniques.

        Returns:
            Dict with response data or error information
        """
        payload = {
            "document_id": str(document_id),
            "query": query,
            "techniques": techniques,
            "query_params": query_params,
        }

        try:
            start_time = time.time()
            response = self.client.post(
                f"{self.server_url}/api/v1/rag/query", json=payload
            )
            elapsed = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "retrieved_chunks": result.get("retrieved_chunks", []),
                    "result_id": result.get("result_id"),
                    "latency_ms": result.get("scores", {}).get("latency_ms", elapsed),
                    "token_count_est": result.get("scores", {}).get(
                        "token_count_est", 0
                    ),
                    "chunks_count": len(result.get("retrieved_chunks", [])),
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
                "detail": f"Request timed out after {self.timeout}s",
            }
        except httpx.RequestError as e:
            return {
                "success": False,
                "error": "Request Error",
                "detail": str(e),
            }
        except Exception as e:
            return {
                "success": False,
                "error": "Unexpected Error",
                "detail": str(e),
            }

    def test_combinations(
        self,
        document_id: UUID,
        query: str,
        combinations_list: List[List[str]],
        query_params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Test all technique combinations.

        Returns:
            List of result dicts
        """
        results = []
        total = len(combinations_list)

        print(f"\nTesting {total} technique combinations...\n")

        for i, techniques in enumerate(combinations_list, 1):
            combo_name = " + ".join(techniques)
            print(f"[{i}/{total}] Testing: {combo_name}...", end=" ", flush=True)

            result = self.execute_query(document_id, query, techniques, query_params)
            result["techniques"] = techniques
            result["combo_name"] = combo_name
            results.append(result)

            if result["success"]:
                print(
                    f"✅ ({result.get('latency_ms', 0):.0f}ms, {result.get('chunks_count', 0)} chunks)"
                )
            else:
                print(f"❌ {result.get('error', 'Error')}")

        return results

    def display_results(
        self, document_path: Path, query: str, results: List[Dict[str, Any]]
    ):
        """Display test results in console."""
        print("\n" + "=" * 80)
        print("RAG Technique Combination Test Results")
        print("=" * 80)
        print(f"\nDocument: {document_path.name}")
        print(f'Query: "{query}"')
        print(f"Server: {self.server_url}\n")

        # Summary table
        print("┌" + "─" * 60 + "┬" + "─" * 8 + "┬" + "─" * 12 + "┬" + "─" * 8 + "┐")
        print(
            f"│ {'Technique Combination':<58} │ {'Status':<6} │ {'Latency':<10} │ {'Chunks':<6} │"
        )
        print("├" + "─" * 60 + "┼" + "─" * 8 + "┼" + "─" * 12 + "┼" + "─" * 8 + "┤")

        for result in results:
            combo = result["combo_name"][:58]
            if result["success"]:
                status = "✅"
                latency = f"{result.get('latency_ms', 0):.0f}ms"
                chunks = str(result.get("chunks_count", 0))
            else:
                status = "❌"
                latency = "-"
                chunks = "-"

            print(f"│ {combo:<58} │ {status:<6} │ {latency:<10} │ {chunks:<6} │")

        print("└" + "─" * 60 + "┴" + "─" * 8 + "┴" + "─" * 12 + "┴" + "─" * 8 + "┘")

        # Detailed results
        print("\n" + "=" * 80)
        print("Detailed Results")
        print("=" * 80 + "\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['combo_name']}")
            if result["success"]:
                response_preview = result.get("response", "")[:200]
                if len(result.get("response", "")) > 200:
                    response_preview += "..."
                print(f"   Response: {response_preview}")
                print(f"   Latency: {result.get('latency_ms', 0):.0f}ms")
                print(f"   Chunks: {result.get('chunks_count', 0)}")
                print(f"   Result ID: {result.get('result_id')}")
            else:
                print(f"   Error: {result.get('error', 'Unknown')}")
                print(f"   Detail: {result.get('detail', 'No details')}")
            print()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Validate document path
    document_path = Path(args.document)
    if not document_path.exists():
        print(f"Error: Document not found: {document_path}", file=sys.stderr)
        sys.exit(1)

    # Generate combinations
    combinations_list = generate_valid_combinations()

    # Initialize tester
    with RAGTester(args.server, timeout=args.timeout) as tester:
        # Check server
        print(f"Checking server connection to {args.server}...")
        if not tester.check_server():
            print(f"Error: Cannot connect to server at {args.server}", file=sys.stderr)
            print("Make sure the server is running.", file=sys.stderr)
            sys.exit(1)
        print("✅ Server connection OK\n")

        # Upload document
        print(f"Uploading document: {document_path.name}...")
        try:
            upload_result = tester.upload_document(document_path, args.session_id)
            document_id = UUID(upload_result["document_id"])
            session_id = upload_result.get("session_id")
            print(f"✅ Document uploaded (ID: {document_id})")
            if session_id:
                print(f"   Session ID: {session_id}")
            print()
        except Exception as e:
            print(f"Error uploading document: {e}", file=sys.stderr)
            sys.exit(1)

        # Prepare query params
        query_params = {
            "top_k": args.top_k,
            "bm25_weight": args.bm25_weight,
            "temperature": args.temperature,
        }

        # Warmup query (optional, helps with model loading)
        if args.warmup:
            print("Running warmup query...")
            try:
                warmup_result = tester.execute_query(
                    document_id, args.query, ["basic_rag"], query_params
                )
                if warmup_result["success"]:
                    print(
                        f"✅ Warmup completed ({warmup_result.get('latency_ms', 0):.0f}ms)\n"
                    )
                else:
                    print(
                        f"⚠️  Warmup failed: {warmup_result.get('error', 'Unknown')}\n"
                    )
            except Exception as e:
                print(f"⚠️  Warmup error: {e}\n")

        # Test combinations
        results = tester.test_combinations(
            document_id, args.query, combinations_list, query_params
        )

        # Display results
        tester.display_results(document_path, args.query, results)

        # Summary with performance stats
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        if successful > 0:
            successful_results = [r for r in results if r["success"]]
            latencies = [r.get("latency_ms", 0) for r in successful_results]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            min_latency = min(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0

            print("=" * 80)
            print("Performance Statistics")
            print("=" * 80)
            print(f"Average latency: {avg_latency:.0f}ms")
            print(f"Min latency: {min_latency:.0f}ms")
            print(f"Max latency: {max_latency:.0f}ms")
            print()

        print("=" * 80)
        print(
            f"Summary: {successful} successful, {failed} failed out of {len(results)} total"
        )
        print("=" * 80)


if __name__ == "__main__":
    main()
