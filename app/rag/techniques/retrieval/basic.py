"""
Basic RAG retriever implementation
"""

import logging
from typing import List, Optional
from uuid import UUID

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.rag.techniques.retrieval.base import BaseRetrieval
from app.services.vectorstore import (
    get_chroma_collection,
    get_parent_document_retriever,
)

logger = logging.getLogger(__name__)


class SafeBasicRetriever(BaseRetriever):
    """
    Wrapper for basic retriever with error handling.

    Wraps a base retriever to handle errors gracefully and ensure
    empty list is returned on failure instead of raising exceptions.
    """

    def __init__(self, base_retriever: BaseRetriever, document_id: UUID):
        """
        Initialize safe basic retriever.

        Args:
            base_retriever: Base retriever to wrap
            document_id: UUID of the document for logging
        """
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "_base_retriever", base_retriever)
        object.__setattr__(self, "_document_id", document_id)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        Retrieve documents with error handling.

        Args:
            query: Query string
            run_manager: Optional callback manager

        Returns:
            List of documents (empty list on error)
        """
        try:
            docs = self._base_retriever.invoke(query)
            return docs if docs else []
        except Exception as e:
            logger.error(
                f"Basic retriever failed for document {self._document_id}: {e}",
                exc_info=True,
            )
            return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        Async retrieve documents with error handling.

        Args:
            query: Query string
            run_manager: Optional callback manager

        Returns:
            List of documents (empty list on error)
        """
        try:
            docs = await self._base_retriever.ainvoke(query)
            return docs if docs else []
        except Exception as e:
            logger.error(
                f"Basic retriever failed for document {self._document_id}: {e}",
                exc_info=True,
            )
            return []


class EmptyRetriever(BaseRetriever):
    """Retriever that always returns empty list."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """Return empty list."""
        return []


class BaseRetrievalAdapter(BaseRetriever):
    """
    Adapter that wraps a BaseRetrieval instance and exposes it as a BaseRetriever.

    This allows BaseRetrieval instances (which have different interface) to be used
    anywhere a BaseRetriever is expected, particularly in orchestration techniques
    like SelfRAG and CRAG.
    """

    def __init__(
        self, base_retrieval: BaseRetrieval, document_id: UUID, top_k: int = 5
    ):
        """
        Initialize the adapter.

        Args:
            base_retrieval: BaseRetrieval instance to wrap
            document_id: UUID of the document to query
            top_k: Number of documents to retrieve
        """
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "_base_retrieval", base_retrieval)
        object.__setattr__(self, "_document_id", document_id)
        object.__setattr__(self, "_top_k", top_k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        Retrieve documents synchronously (not typically used).

        Args:
            query: Query string
            run_manager: Optional callback manager

        Returns:
            List of documents
        """
        logger.warning("Synchronous retrieval not efficiently supported in adapter")
        return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        Retrieve documents asynchronously.

        Args:
            query: Query string
            run_manager: Optional callback manager

        Returns:
            List of documents
        """
        try:
            docs = await self._base_retrieval.retrieve(
                query=query,
                document_id=self._document_id,
                top_k=self._top_k,
            )
            return docs if docs else []
        except Exception as e:
            logger.error(
                f"Adapter retrieval failed for document {self._document_id}: {e}",
                exc_info=True,
            )
            return []


class BasicRetrieval(BaseRetrieval):
    """
    Basic vector retrieval technique.

    Uses vector similarity search via ChromaDB or parent document retrieval
    if the parent_document chunking strategy is configured.
    """

    def __init__(
        self,
        document_id: UUID,
        top_k: int = 5,
        chunking_strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize basic retrieval.

        Args:
            document_id: UUID of the document to query
            top_k: Number of documents to retrieve
            chunking_strategy: Optional chunking strategy (if "parent_document", uses ParentDocumentRetriever)
            chunk_size: Optional chunk size (required if using parent_document)
            chunk_overlap: Optional chunk overlap (required if using parent_document)
            **kwargs: Additional arguments
        """
        self.document_id = document_id
        self.top_k = top_k
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _create_retriever(self) -> BaseRetriever:
        """Create the underlying retriever instance."""
        try:
            # Use ParentDocumentRetriever if chunking strategy is parent_document
            if self.chunking_strategy == "parent_document":
                if self.chunk_size is None or self.chunk_overlap is None:
                    logger.warning(
                        f"Parent document retriever requires chunk_size and chunk_overlap. "
                        f"Falling back to standard retriever for document {self.document_id}"
                    )
                else:
                    try:
                        parent_retriever = get_parent_document_retriever(
                            self.document_id,
                            self.chunk_size,
                            self.chunk_overlap,
                            self.top_k,
                        )
                        # Wrap in SafeBasicRetriever for consistent error handling
                        return SafeBasicRetriever(parent_retriever, self.document_id)
                    except Exception as e:
                        logger.error(
                            f"Failed to create parent document retriever for document {self.document_id}: {e}",
                            exc_info=True,
                        )
                        # Fall back to standard retriever

            # Standard vector retriever
            collection = get_chroma_collection(self.document_id)
            base_retriever = collection.as_retriever(search_kwargs={"k": self.top_k})

            # Wrap the retriever to handle errors gracefully
            return SafeBasicRetriever(base_retriever, self.document_id)
        except Exception as e:
            logger.error(
                f"Failed to create basic retriever for document {self.document_id}: {e}",
                exc_info=True,
            )
            # Return a retriever that always returns empty list
            return EmptyRetriever()

    async def retrieve(
        self,
        query: str,
        document_id: UUID,
        top_k: int = 5,
        **kwargs,
    ) -> List[Document]:
        """
        Retrieve relevant documents.

        Args:
            query: User query
            document_id: UUID of the document to query (overrides init value if provided)
            top_k: Number of documents to retrieve (overrides init value if provided)
            **kwargs: Additional parameters

        Returns:
            List of retrieved Document objects
        """
        # Use provided document_id and top_k, or fall back to instance values
        effective_document_id = (
            document_id if document_id != self.document_id else self.document_id
        )
        effective_top_k = top_k if top_k != self.top_k else self.top_k

        retriever = self._create_retriever()
        return await retriever.ainvoke(query)
