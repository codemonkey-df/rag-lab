"""
Verification utilities for indexing strategy tests.

Provides helper functions to verify strategy-specific characteristics
of chunks retrieved from ChromaDB.
"""

from typing import Any, Dict, List


def verify_headers_strategy(
    chunk_samples: List[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verify headers strategy characteristics.

    Headers should be:
    - Prepended to chunk content as [header]\n\n{content}
    - Stored in metadata as metadata["header"]

    Args:
        chunk_samples: Sample chunks from collection
        config: Configuration used

    Returns:
        Dictionary with verification results
    """
    verification = {
        "strategy_applied": True,
        "issues": [],
        "evidence": [],
    }

    if not chunk_samples:
        verification["strategy_applied"] = False
        verification["issues"].append("No chunks found to verify")
        return verification

    has_headers = False
    chunks_checked = 0
    chunks_with_header_text = 0
    chunks_with_header_metadata = 0

    for chunk in chunk_samples:
        chunks_checked += 1
        # Check full text first (for verification), then truncated text (for display)
        full_text = chunk.get("full_text", "")
        text = chunk.get("text", "")  # First 200 chars
        metadata = chunk.get("metadata", {})

        # Check for single bracket pattern [header]\n\n or metadata
        # FIXED: Original test looked for [[ or ]] (double brackets)
        # Implementation uses single brackets: [header]\n\n{content}
        # Check full_text first (more reliable), then text as fallback
        text_to_check = full_text if full_text else text

        # More flexible pattern matching - header might have newlines
        header_in_text = False
        if text_to_check:
            # Check if starts with [ and has ]\n\n pattern
            if text_to_check.startswith("[") and "]\n\n" in text_to_check:
                header_in_text = True
                chunks_with_header_text += 1
            # Also check for ]\n\n anywhere (in case of whitespace)
            elif "]\n\n" in text_to_check and text_to_check.find("]\n\n") < 100:
                # Header might be after some whitespace
                header_in_text = True
                chunks_with_header_text += 1

        header_in_metadata = metadata.get("header") is not None
        if header_in_metadata:
            chunks_with_header_metadata += 1

        if header_in_text or header_in_metadata:
            has_headers = True
            header_value = metadata.get("header", "extracted from text")
            # Extract header from text if not in metadata
            if not header_in_metadata and header_in_text:
                try:
                    header_end = text_to_check.find("]\n\n")
                    if header_end > 0:
                        header_value = text_to_check[1:header_end]
                except Exception:
                    pass
            verification["evidence"].append(
                f"Found header in chunk {chunks_checked}: {header_value[:50]}..."
            )
            # Don't break - continue checking to get statistics

    if not has_headers:
        # Debug: show what we actually found
        sample_texts = [
            chunk.get("full_text", chunk.get("text", ""))[:150]
            for chunk in chunk_samples[:3]
        ]
        sample_metadata_keys = [
            list(chunk.get("metadata", {}).keys()) for chunk in chunk_samples[:3]
        ]
        verification["issues"].append(
            f"Headers strategy: No headers found in {chunks_checked} sample chunks. "
            f"Expected format: [header]\\n\\n{{content}} or metadata['header']. "
            f"Checked {chunks_checked} chunks. "
            f"Sample text starts: {[t[:80] + '...' if len(t) > 80 else t for t in sample_texts]}. "
            f"Sample metadata keys: {sample_metadata_keys}. "
            f"This might indicate: (1) LLM not available/running, (2) Header generation failed silently, "
            f"(3) Headers not preserved in ChromaDB, or (4) Background task not completed."
        )
        verification["strategy_applied"] = False
    else:
        # Add statistics
        verification["evidence"].append(
            f"Found headers in {chunks_with_header_text + chunks_with_header_metadata} of {chunks_checked} chunks "
            f"({chunks_with_header_text} in text, {chunks_with_header_metadata} in metadata)"
        )

    return verification


def verify_semantic_strategy(
    chunk_samples: List[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verify semantic strategy characteristics.

    Semantic chunks should have variable sizes (not fixed like standard).

    Args:
        chunk_samples: Sample chunks from collection
        config: Configuration used

    Returns:
        Dictionary with verification results
    """
    verification = {
        "strategy_applied": True,
        "issues": [],
        "evidence": [],
    }

    if not chunk_samples:
        verification["strategy_applied"] = False
        verification["issues"].append("No chunks found to verify")
        return verification

    text_lengths = [chunk.get("text_length", 0) for chunk in chunk_samples]
    if text_lengths:
        length_variance = max(text_lengths) - min(text_lengths)
        expected_size = config.get("chunk_size", 1024)
        # Semantic chunks should vary more than 20% from expected
        if length_variance < expected_size * 0.2:
            verification["issues"].append(
                f"Semantic chunks seem too uniform (variance: {length_variance}, "
                f"expected variance > {expected_size * 0.2})"
            )
        else:
            verification["evidence"].append(
                f"Semantic chunk size variance: {length_variance} (good)"
            )

    return verification


def verify_proposition_strategy(
    chunk_samples: List[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verify proposition strategy characteristics.

    Proposition chunks should be shorter and more atomic (facts).

    Args:
        chunk_samples: Sample chunks from collection
        config: Configuration used

    Returns:
        Dictionary with verification results
    """
    verification = {
        "strategy_applied": True,
        "issues": [],
        "evidence": [],
    }

    if not chunk_samples:
        verification["strategy_applied"] = False
        verification["issues"].append("No chunks found to verify")
        return verification

    text_lengths = [chunk.get("text_length", 0) for chunk in chunk_samples]
    if text_lengths:
        avg_length = sum(text_lengths) / len(text_lengths)
        # Propositions should be relatively short (atomic facts)
        if avg_length > 500:
            verification["issues"].append(
                f"Proposition chunks seem too long (avg: {avg_length:.0f} chars, "
                "expected < 500 for atomic facts)"
            )
        else:
            verification["evidence"].append(
                f"Proposition chunks are appropriately short (avg: {avg_length:.0f} chars)"
            )

    return verification


def verify_parent_document_strategy(
    chunk_samples: List[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verify parent document strategy characteristics.

    Parent document should have parent_id in metadata for child chunks.

    Args:
        chunk_samples: Sample chunks from collection
        config: Configuration used

    Returns:
        Dictionary with verification results
    """
    verification = {
        "strategy_applied": True,
        "issues": [],
        "evidence": [],
    }

    if not chunk_samples:
        verification["strategy_applied"] = False
        verification["issues"].append("No chunks found to verify")
        return verification

    # Parent document should have parent_id in metadata for child chunks
    for chunk in chunk_samples:
        metadata = chunk.get("metadata", {})
        if metadata.get("parent_id") or metadata.get("parent"):
            verification["evidence"].append(
                "Found parent_id in chunk metadata (parent_document strategy)"
            )
            break
    # Note: parent_document might not always show parent_id in samples
    # This is a soft check - if we don't find it, we don't fail

    return verification


def verify_standard_strategy(
    chunk_samples: List[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verify standard strategy characteristics.

    Standard chunks should have consistent sizes close to chunk_size.

    Args:
        chunk_samples: Sample chunks from collection
        config: Configuration used

    Returns:
        Dictionary with verification results
    """
    verification = {
        "strategy_applied": True,
        "issues": [],
        "evidence": [],
    }

    if not chunk_samples:
        verification["strategy_applied"] = False
        verification["issues"].append("No chunks found to verify")
        return verification

    text_lengths = [chunk.get("text_length", 0) for chunk in chunk_samples]
    if text_lengths:
        expected_size = config.get("chunk_size", 1024)
        avg_length = sum(text_lengths) / len(text_lengths)
        # Standard chunks should be close to chunk_size
        if abs(avg_length - expected_size) > expected_size * 0.5:
            verification["issues"].append(
                f"Standard chunks size mismatch (avg: {avg_length:.0f}, "
                f"expected: {expected_size}, tolerance: ±{expected_size * 0.5})"
            )
        else:
            verification["evidence"].append(
                f"Standard chunks have expected size (avg: {avg_length:.0f})"
            )

    return verification


def get_chunk_samples(
    collection_data: Dict[str, Any], sample_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Sample chunks from collection data for verification.

    Args:
        collection_data: Data from collection.get() with keys:
                        - documents: List of text content
                        - metadatas: List of metadata dicts
                        - ids: List of chunk IDs
        sample_size: Number of chunks to sample

    Returns:
        List of sample chunk dictionaries with:
        - id: Chunk ID
        - text: Chunk text content (first 200 chars for display)
        - full_text: Full chunk text content (for verification)
        - text_length: Full text length
        - metadata: Chunk metadata
    """
    documents = collection_data.get("documents", [])
    metadatas = collection_data.get("metadatas", [])
    ids = collection_data.get("ids", [])

    samples = []
    for i in range(min(sample_size, len(documents))):
        full_text = documents[i] if documents[i] else ""
        samples.append(
            {
                "id": ids[i] if i < len(ids) else None,
                "text": full_text[:200],  # First 200 chars for display
                "full_text": full_text,  # Full text for verification
                "text_length": len(full_text),
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }
        )
    return samples
