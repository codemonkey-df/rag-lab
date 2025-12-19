"""
Ollama health check functionality
"""
import logging
import httpx
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from app.core.config import get_settings

logger = logging.getLogger(__name__)


async def check_ollama_health() -> tuple[bool, str]:
    """
    Check if Ollama API is accessible and models are available.
    
    Returns:
        tuple[bool, str]: (is_healthy, message)
    """
    settings = get_settings()
    
    try:
        # Check API accessibility
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            if response.status_code != 200:
                return False, f"Ollama API returned status {response.status_code}"
        
        # Check LLM model
        try:
            llm = OllamaLLM(
                model=settings.ollama_llm_model,
                base_url=settings.ollama_base_url
            )
            await llm.ainvoke("test")
        except Exception as e:
            return False, f"LLM model '{settings.ollama_llm_model}' not available: {str(e)}"
        
        # Check embedding model
        try:
            embeddings = OllamaEmbeddings(
                model=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url
            )
            await embeddings.aembed_query("test")
        except Exception as e:
            return False, f"Embedding model '{settings.ollama_embedding_model}' not available: {str(e)}"
        
        return True, "Ollama health check passed"
        
    except httpx.ConnectError:
        return False, f"Cannot connect to Ollama at {settings.ollama_base_url}. Ensure 'ollama serve' is running."
    except Exception as e:
        return False, f"Ollama health check failed: {str(e)}"


def get_ollama_setup_instructions() -> str:
    """Return setup instructions for Ollama"""
    return """
    Ollama Setup Required:
    1. Install Ollama: https://ollama.ai
    2. Start Ollama service: ollama serve
    3. Pull required models:
       ollama pull nomic-embed-text
       ollama pull llama3.2
    """
