"""
RAG Techniques organized by pipeline stage

Categories:
- indexing: Indexing (document chunking) techniques
- retrieval: Retrieval Method (engine) techniques
- query_expansion: Query Expansion (enrichment) techniques
- filtering: Filtering and post-processing (reranking, compression) techniques
- orchestration: Orchestration/Control techniques
"""

from app.rag.techniques.filtering import (
    Reranker,
    compress_documents,
)
from app.rag.techniques.indexing import (
    BaseIndexingStrategy,
    HeadersStrategy,
    IndexingStrategyFactory,
    ParentDocumentStrategy,
    PropositionStrategy,
    SemanticStrategy,
    StandardStrategy,
)
from app.rag.techniques.orchestration import (
    CRAG,
    AdaptiveRetrieval,
    SelfRAG,
)
from app.rag.techniques.query_expansion import (
    expand_query_with_hyde,
)
from app.rag.techniques.retrieval import (
    BasicRetrieval,
    HybridRetrieval,
)

__all__ = [
    # Indexing
    "BaseIndexingStrategy",
    "IndexingStrategyFactory",
    "StandardStrategy",
    "ParentDocumentStrategy",
    "SemanticStrategy",
    "HeadersStrategy",
    "PropositionStrategy",
    # Retrieval
    "BasicRetrieval",
    "HybridRetrieval",
    # Query Expansion
    "expand_query_with_hyde",
    # Filtering
    "compress_documents",
    "Reranker",
    # Orchestration
    "AdaptiveRetrieval",
    "CRAG",
    "SelfRAG",
]
