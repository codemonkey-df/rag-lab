"""
BM25 retriever management with on-demand rebuild from ChromaDB
"""
from uuid import UUID
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from app.services.vectorstore import get_chroma_collection


class BM25Manager:
    """
    Manages BM25Retrievers rebuilt from ChromaDB on-demand.
    Acts as a Factory and LRU Cache.
    
    Strategy:
    - No persistence (simpler, avoids versioning)
    - Rebuild from ChromaDB chunks when needed
    - LRU cache to avoid repeated rebuilds
    - Returns fully configured LangChain BM25Retriever
    """
    
    def __init__(self):
        # Cache stores the actual Retriever object
        self._cache: dict[str, BM25Retriever] = {}
    
    def get_retriever(self, document_id: UUID) -> BM25Retriever:
        """
        Get a BM25Retriever for a document, rebuilding from ChromaDB if needed.
        
        Uses in-memory cache to avoid repeated rebuilds.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            BM25Retriever instance
        """
        cache_key = str(document_id)
        
        # 1. Check Cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 2. Rebuild from ChromaDB (The "On-Demand" Logic)
        collection = get_chroma_collection(document_id)
        
        # Fetch ALL data from Chroma for this document
        # Note: Chroma.get() returns dict with 'ids', 'documents', 'metadatas'
        data = collection.get()
        
        if not data or not data.get('ids'):
            # Return empty retriever if no docs found
            return BM25Retriever.from_documents([], k=5)
        
        # Reconstruct Document objects for LangChain
        docs = []
        texts = data.get('documents', [])
        metadatas = data.get('metadatas', [])
        
        for i, text in enumerate(texts):
            # Ensure metadata is valid dict
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            docs.append(Document(page_content=text, metadata=meta))
        
        # 3. Create LangChain BM25 Retriever
        retriever = BM25Retriever.from_documents(docs, k=5)
        
        # 4. Cache and Return
        self._cache[cache_key] = retriever
        return retriever
    
    def clear_cache(self, document_id: UUID | None = None):
        """
        Clear BM25 cache for a document or all documents.
        
        Args:
            document_id: UUID of the document to clear, or None to clear all
        """
        if document_id:
            cache_key = str(document_id)
            self._cache.pop(cache_key, None)
        else:
            self._cache.clear()


# Global singleton
bm25_manager = BM25Manager()
