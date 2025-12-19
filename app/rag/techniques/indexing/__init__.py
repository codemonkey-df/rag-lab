"""
Indexing techniques and strategies for document processing.

This module provides all available indexing strategies that can be used
to ingest documents into the RAG system.

Available Strategies:
- standard: Fast baseline RecursiveCharacterTextSplitter
- parent_document: Small child chunks + large parent chunks
- semantic: Embedding-based semantic chunking
- headers: Headers + standard/semantic chunking
- proposition: Atomic fact extraction with LLM
"""

# Core interfaces and factory
from app.rag.techniques.indexing.base import BaseIndexingStrategy
from app.rag.techniques.indexing.factory import IndexingStrategyFactory

# Concrete strategy implementations
from app.rag.techniques.indexing.headers import HeadersStrategy
from app.rag.techniques.indexing.parent import ParentDocumentStrategy
from app.rag.techniques.indexing.proposition import PropositionStrategy
from app.rag.techniques.indexing.semantic import SemanticStrategy
from app.rag.techniques.indexing.standard import StandardStrategy

# Utilities
from app.rag.techniques.indexing.utils import load_pdf_with_metadata

__all__ = [
    # Base and factory
    "BaseIndexingStrategy",
    "IndexingStrategyFactory",
    # Strategies
    "StandardStrategy",
    "ParentDocumentStrategy",
    "SemanticStrategy",
    "HeadersStrategy",
    "PropositionStrategy",
    # Utilities
    "load_pdf_with_metadata",
]
