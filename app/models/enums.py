"""
Enumerations for RAG techniques and strategies
"""
from enum import Enum


class RAGTechnique(str, Enum):
    """RAG techniques organized by layer."""
    
    # Layer 1: Indexing Strategy (Mutually Exclusive)
    STANDARD_CHUNKING = "standard_chunking"
    PARENT_DOCUMENT = "parent_document"
    SEMANTIC_CHUNKING = "semantic_chunking"
    CONTEXTUAL_HEADERS = "contextual_headers"
    PROPOSITION_CHUNKING = "proposition_chunking"
    
    # Layer 2: Pipeline Components (Mix & Match)
    HYDE = "hyde"
    BASIC_RAG = "basic_rag"
    FUSION_RETRIEVAL = "fusion_retrieval"
    RERANKING = "reranking"
    CONTEXTUAL_COMPRESSION = "contextual_compression"
    
    # Layer 3: Advanced Controllers (Max One)
    SELF_RAG = "self_rag"
    CRAG = "crag"
    ADAPTIVE_RETRIEVAL = "adaptive_retrieval"


# Layer mappings
LAYER_1_TECHNIQUES = {
    RAGTechnique.STANDARD_CHUNKING,
    RAGTechnique.PARENT_DOCUMENT,
    RAGTechnique.SEMANTIC_CHUNKING,
    RAGTechnique.CONTEXTUAL_HEADERS,
    RAGTechnique.PROPOSITION_CHUNKING,
}

LAYER_2_TECHNIQUES = {
    RAGTechnique.HYDE,
    RAGTechnique.BASIC_RAG,
    RAGTechnique.FUSION_RETRIEVAL,
    RAGTechnique.RERANKING,
    RAGTechnique.CONTEXTUAL_COMPRESSION,
}

LAYER_3_TECHNIQUES = {
    RAGTechnique.SELF_RAG,
    RAGTechnique.CRAG,
    RAGTechnique.ADAPTIVE_RETRIEVAL,
}
