"""
Configuration management using pydantic-settings
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Database
    database_url: str = "sqlite:///playground.db"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "gpt-oss:20b-cloud"  # "llama3.2"
    ollama_hyde_model: str = "gpt-oss:20b-cloud"  # Smaller model for HyDE (faster)
    ollama_embedding_model: str = (
        "nomic-embed-text"  # "jeffh/intfloat-multilingual-e5-large:q8_0"
    )

    # Embeddings
    embedding_dimension: int = (
        1024
        if ollama_embedding_model == "jeffh/intfloat-multilingual-e5-large:q8_0"
        else 768
    )

    # Chroma
    chroma_persist_dir: str = "./chromadb"

    # File Upload
    max_file_size_mb: int = 50
    upload_dir: str = "./uploads"

    # Web Search
    tavily_api_key: str | None = None
    use_tavily: bool = True

    # Logging
    log_level: str = "INFO"

    # RAG Parameters
    max_context_tokens: int = 4096  # Default for llama3.2
    default_top_k: int = 5
    default_bm25_weight: float = 0.5
    default_temperature: float = 0.7
    hyde_retrieval_multiplier: int = 3  # Retrieve 3x more docs when HyDE enabled

    # Chunking Defaults
    default_chunk_size: int = 1024
    default_chunk_overlap: int = 200

    # Proposition Chunking Defaults
    proposition_initial_chunk_size: int = (
        200  # Initial chunk size for proposition generation
    )
    proposition_chunk_overlap: int = 50  # Overlap for initial chunking
    proposition_quality_threshold_accuracy: int = 7  # Minimum accuracy score (1-10)
    proposition_quality_threshold_clarity: int = 7  # Minimum clarity score (1-10)
    proposition_quality_threshold_completeness: int = (
        7  # Minimum completeness score (1-10)
    )
    proposition_quality_threshold_conciseness: int = (
        7  # Minimum conciseness score (1-10)
    )

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
