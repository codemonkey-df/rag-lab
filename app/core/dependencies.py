"""
Dependency injection for shared resources
"""
import logging
from functools import lru_cache
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from chromadb import PersistentClient
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@lru_cache()
def get_llm() -> OllamaLLM:
    """Get Ollama LLM instance"""
    return OllamaLLM(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.7
    )


@lru_cache()
def get_hyde_llm() -> OllamaLLM:
    """
    Get smaller Ollama LLM instance for HyDE (faster, less accurate).
    
    Optimized for speed:
    - num_predict=150: Reduced from 200 for faster generation
    - num_ctx=1024: Limit context window for speed
    """
    llm = OllamaLLM(
        model=settings.ollama_hyde_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
        num_predict=150,  # Reduced from 200 for faster generation
        num_ctx=1024      # Limit context window for speed
    )
    logger.info(f"HyDE LLM configured: model={settings.ollama_hyde_model}, num_predict=150, num_ctx=1024")
    return llm


@lru_cache()
def get_embeddings() -> OllamaEmbeddings:
    """Get Ollama embeddings instance"""
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url
    )


@lru_cache()
def get_embedding_dimension() -> int:
    """
    Get the actual embedding dimension from the embedding model.
    
    This function tests the embedding model to determine its actual dimension,
    which may differ from the config value. The result is cached to avoid
    repeated API calls.
    
    Returns:
        The actual embedding dimension produced by the model
    """
    try:
        embeddings = get_embeddings()
        # Test with a small string to get the dimension
        test_embedding = embeddings.embed_query("test")
        return len(test_embedding)
    except Exception as e:
        # Fallback to config value if detection fails
        return settings.embedding_dimension


@lru_cache()
def get_chroma_client() -> PersistentClient:
    """Get ChromaDB persistent client"""
    return PersistentClient(path=settings.chroma_persist_dir)
