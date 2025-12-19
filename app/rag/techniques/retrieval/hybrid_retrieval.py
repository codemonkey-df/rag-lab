"""
Hybrid retrieval implementation (BM25 + Dense Vector)
"""

import logging
from typing import List, Optional
from uuid import UUID

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.rag.techniques.retrieval.base import BaseRetrieval
from app.services.bm25_manager import bm25_manager
from app.services.vectorstore import (
    get_chroma_collection,
    get_parent_document_retriever,
)

logger = logging.getLogger(__name__)


class LangChainHybridRetriever(BaseRetriever):
    """
    Custom hybrid retriever that combines vector and BM25 retrievers with weighted fusion.

    This is a simple ensemble that:
    1. Retrieves from both retrievers
    2. Combines results with reciprocal rank fusion (RRF)
    3. Returns top_k results

    Supports parent document retrieval: if chunking_strategy is "parent_document",
    the vector retriever will use ParentDocumentRetriever to return parent documents
    instead of child chunks.
    """

    document_id: UUID
    top_k: int = 5
    bm25_weight: float = 0.5
    chunking_strategy: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """Retrieve documents using hybrid approach."""
        try:
            # 1. Vector Retriever (Dense)
            # Use ParentDocumentRetriever if chunking strategy is parent_document
            try:
                if (
                    self.chunking_strategy == "parent_document"
                    and self.chunk_size is not None
                    and self.chunk_overlap is not None
                ):
                    # Use ParentDocumentRetriever to get parent documents
                    parent_retriever = get_parent_document_retriever(
                        self.document_id,
                        self.chunk_size,
                        self.chunk_overlap,
                        self.top_k * 2,
                    )
                    vector_docs = parent_retriever.invoke(query) or []
                else:
                    # Standard vector retriever
                    vectorstore = get_chroma_collection(self.document_id)
                    vector_retriever = vectorstore.as_retriever(
                        search_kwargs={"k": self.top_k * 2}
                    )
                    vector_docs = vector_retriever.invoke(query) or []
            except Exception as e:
                error_msg = str(e)
                if "dimension" in error_msg.lower():
                    logger.error(
                        f"Vector retrieval failed due to embedding dimension mismatch: {e}. "
                        f"This usually means the embedding model was changed. "
                        f"Please restart the application and re-index documents if needed."
                    )
                else:
                    logger.warning(f"Vector retrieval failed: {e}")
                vector_docs = []

            # 2. BM25 Retriever (Sparse)
            try:
                bm25_retriever = bm25_manager.get_retriever(self.document_id)
                bm25_retriever.k = self.top_k * 2  # Get more candidates for fusion
                bm25_docs = bm25_retriever.invoke(query) or []
            except Exception as e:
                logger.warning(f"BM25 retrieval failed: {e}")
                bm25_docs = []

            # If both retrievers failed, return empty list
            if not vector_docs and not bm25_docs:
                logger.warning(
                    f"No documents retrieved from either retriever for query: {query[:50]}"
                )
                return []

            # If only one retriever succeeded, return its results
            if not vector_docs:
                return bm25_docs[: self.top_k]
            if not bm25_docs:
                return vector_docs[: self.top_k]

            # 3. Reciprocal Rank Fusion (RRF)
            # Score each document based on its rank in each retriever
            doc_scores = {}

            # Score vector results
            for rank, doc in enumerate(vector_docs, start=1):
                try:
                    doc_id = (
                        doc.page_content if hasattr(doc, "page_content") else str(doc)
                    )
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {"doc": doc, "score": 0.0}
                    # RRF score with weight
                    doc_scores[doc_id]["score"] += (1.0 - self.bm25_weight) * (
                        1.0 / (rank + 60)
                    )
                except Exception as e:
                    logger.warning(f"Error scoring vector document: {e}")
                    continue

            # Score BM25 results
            for rank, doc in enumerate(bm25_docs, start=1):
                try:
                    doc_id = (
                        doc.page_content if hasattr(doc, "page_content") else str(doc)
                    )
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {"doc": doc, "score": 0.0}
                    # RRF score with weight
                    doc_scores[doc_id]["score"] += self.bm25_weight * (
                        1.0 / (rank + 60)
                    )
                except Exception as e:
                    logger.warning(f"Error scoring BM25 document: {e}")
                    continue

            # 4. Sort by combined score and return top_k
            sorted_docs = sorted(
                doc_scores.values(), key=lambda x: x["score"], reverse=True
            )
            results = [item["doc"] for item in sorted_docs[: self.top_k]]

            return results if results else []
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}", exc_info=True)
            return []


class HybridRetrieval(BaseRetrieval):
    """
    Hybrid retrieval technique combining BM25 (sparse) and vector (dense) search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from both retrieval methods.
    """

    def __init__(
        self,
        document_id: UUID,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        chunking_strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize hybrid retrieval.

        Args:
            document_id: UUID of the document to query
            top_k: Number of documents to retrieve
            bm25_weight: Weight for BM25 (0.0 to 1.0), vector weight is (1.0 - bm25_weight)
            chunking_strategy: Optional chunking strategy (if "parent_document", uses ParentDocumentRetriever)
            chunk_size: Optional chunk size (required if using parent_document)
            chunk_overlap: Optional chunk overlap (required if using parent_document)
            **kwargs: Additional arguments
        """
        self.document_id = document_id
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _create_retriever(self) -> BaseRetriever:
        """Create the underlying hybrid retriever instance."""
        return LangChainHybridRetriever(
            document_id=self.document_id,
            top_k=self.top_k,
            bm25_weight=self.bm25_weight,
            chunking_strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    async def retrieve(
        self,
        query: str,
        document_id: UUID,
        top_k: int = 5,
        **kwargs,
    ) -> List[Document]:
        """
        Retrieve relevant documents using hybrid approach.

        Args:
            query: User query
            document_id: UUID of the document to query (overrides init value if provided)
            top_k: Number of documents to retrieve (overrides init value if provided)
            **kwargs: Additional parameters (bm25_weight can be overridden)

        Returns:
            List of retrieved Document objects
        """
        # Use provided parameters or fall back to instance values
        effective_top_k = top_k if top_k != self.top_k else self.top_k
        effective_bm25_weight = kwargs.get("bm25_weight", self.bm25_weight)

        retriever = LangChainHybridRetriever(
            document_id=document_id,
            top_k=effective_top_k,
            bm25_weight=effective_bm25_weight,
            chunking_strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return await retriever.ainvoke(query)
