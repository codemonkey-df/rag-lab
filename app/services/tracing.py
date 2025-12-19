"""
Tracing and chunk source tracking with enhanced metadata capture
"""
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def capture_retrieved_chunks(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Capture retrieved chunks with metadata for tracing.
    
    Args:
        documents: List of retrieved documents
        
    Returns:
        List of chunk dicts with full metadata
    """
    if not documents:
        return []
    
    chunks = []
    for doc in documents:
        try:
            # Safely access document attributes
            page_content = getattr(doc, 'page_content', '') or ''
            metadata = getattr(doc, 'metadata', {}) or {}
            
            chunk = {
                "text": page_content,
                "page": metadata.get("page"),
                "line_start": metadata.get("line_start"),
                "line_end": metadata.get("line_end"),
                "score": metadata.get("score"),
                "header": metadata.get("header"),
            }
            chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Error capturing chunk metadata: {e}")
            # Add minimal chunk info
            chunks.append({
                "text": str(doc) if doc else "",
                "page": None,
                "line_start": None,
                "line_end": None,
                "score": None,
            })
    
    return chunks


def capture_pipeline_metadata(
    techniques: List[str],
    retrieved_chunks: List[Document],
    additional_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Capture full pipeline metadata for complex flows.
    
    Args:
        techniques: List of technique names used
        retrieved_chunks: Retrieved documents
        additional_metadata: Extra metadata from advanced techniques
        
    Returns:
        Comprehensive metadata dict
    """
    metadata = {
        "techniques": techniques,
        "chunks_retrieved": len(retrieved_chunks),
        "chunks": capture_retrieved_chunks(retrieved_chunks),
    }
    
    if additional_metadata:
        metadata.update(additional_metadata)
    
    return metadata


def capture_self_rag_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture Self-RAG specific metadata.
    
    Args:
        result: Self-RAG result dict
        
    Returns:
        Formatted metadata for storage
    """
    try:
        metadata = result.get("metadata", {})
        
        # Safely capture chunks
        documents = result.get("documents", [])
        chunks = []
        if documents:
            try:
                chunks = capture_retrieved_chunks(documents)
            except Exception as e:
                logger.warning(f"Error capturing Self-RAG chunks: {e}")
                chunks = []
        
        return {
            "self_rag": {
                "retrieved": result.get("retrieved", False),
                "relevance_score": result.get("relevance_score"),
                "critique": result.get("critique"),
                "retrieval_decision": metadata.get("retrieval_decision"),
                "relevance_evaluation": metadata.get("relevance_evaluation"),
            },
            "chunks": chunks
        }
    except Exception as e:
        logger.error(f"Error capturing Self-RAG metadata: {e}", exc_info=True)
        return {"chunks": []}


def capture_crag_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture CRAG specific metadata.
    
    Args:
        result: CRAG result dict
        
    Returns:
        Formatted metadata for storage
    """
    try:
        metadata = result.get("metadata", {})
        
        crag_meta = {
            "crag": {
                "source": result.get("source"),
                "relevance_score": result.get("relevance_score"),
                "web_search_triggered": metadata.get("web_search_triggered"),
                "relevance_evaluation": metadata.get("relevance_evaluation"),
            }
        }
        
        # Add chunks based on source
        source = result.get("source")
        if source == "documents" and result.get("documents"):
            try:
                crag_meta["chunks"] = capture_retrieved_chunks(result.get("documents"))
            except Exception as e:
                logger.warning(f"Error capturing CRAG document chunks: {e}")
                crag_meta["chunks"] = []
        elif source == "web" and result.get("web_results"):
            crag_meta["web_results"] = result.get("web_results")
        else:
            crag_meta["chunks"] = []
        
        return crag_meta
    except Exception as e:
        logger.error(f"Error capturing CRAG metadata: {e}", exc_info=True)
        return {"chunks": []}


def capture_adaptive_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture Adaptive Retrieval specific metadata.
    
    Args:
        result: Adaptive Retrieval result dict
        
    Returns:
        Formatted metadata for storage
    """
    try:
        documents = result.get("documents", [])
        chunks = []
        if documents:
            try:
                chunks = capture_retrieved_chunks(documents)
            except Exception as e:
                logger.warning(f"Error capturing Adaptive Retrieval chunks: {e}")
                chunks = []
        
        return {
            "adaptive": {
                "strategy_used": result.get("strategy_used"),
                "classification": result.get("classification"),
                "metadata": result.get("metadata"),
            },
            "chunks": chunks
        }
    except Exception as e:
        logger.error(f"Error capturing Adaptive Retrieval metadata: {e}", exc_info=True)
        return {"chunks": []}
