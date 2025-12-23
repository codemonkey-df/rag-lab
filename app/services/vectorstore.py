"""
Vector store and document storage management
"""

from uuid import UUID

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.dependencies import get_chroma_client, get_embeddings

settings = get_settings()


def get_chroma_collection(document_id: UUID) -> Chroma:
    """
    Get or create ChromaDB collection for a document.

    Args:
        document_id: UUID of the document

    Returns:
        Chroma instance for the document
    """
    client = get_chroma_client()
    embeddings = get_embeddings()

    collection_name = f"doc_{document_id}"

    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def get_parent_document_store(document_id: UUID) -> InMemoryStore:
    """
    Get in-memory document store for parent document storage.

    Note: This is a sandbox/experimental platform, so data is lost on restart.
    Each document gets its own InMemoryStore instance to prevent key collisions.

    Args:
        document_id: UUID of the document (for isolation, though not strictly needed with InMemoryStore)

    Returns:
        InMemoryStore instance
    """
    return InMemoryStore()


def create_collection_for_document(document_id: UUID) -> Chroma:
    """
    Create a new ChromaDB collection for a document.

    Args:
        document_id: UUID of the document

    Returns:
        Chroma instance for the document
    """
    return get_chroma_collection(document_id)


def get_parent_document_retriever(
    document_id: UUID,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    top_k: int = 5,
) -> ParentDocumentRetriever:
    """
    Get or create ParentDocumentRetriever for a document.

    This retriever will:
    1. Retrieve child chunks from vectorstore (for precise matching)
    2. Fetch parent documents from docstore (for richer context)
    3. Return parent documents instead of child chunks

    Args:
        document_id: UUID of the document
        chunk_size: Child chunk size (should match ingestion settings)
        chunk_overlap: Child chunk overlap (should match ingestion settings)
        top_k: Number of documents to retrieve

    Returns:
        ParentDocumentRetriever instance configured for the document
    """
    vectorstore = get_chroma_collection(document_id)
    docstore = get_parent_document_store(document_id)

    # Create child splitter (must match ingestion settings)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Create parent splitter (must match ingestion settings)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,
        chunk_overlap=0,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": top_k},
    )

    return retriever


def delete_collection_for_document(document_id: UUID) -> bool:
    """
    Delete ChromaDB collection for a document.

    Note: Parent document store is in-memory, so no cleanup needed.

    Args:
        document_id: UUID of the document

    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_chroma_client()
        collection_name = f"doc_{document_id}"
        client.delete_collection(collection_name)
        return True
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return False
