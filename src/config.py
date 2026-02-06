"""
RAG Pipeline Configuration.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    
    llm_provider: Literal["openai", "anthropic"] = Field(default="openai")
    llm_model: str = Field(default="gpt-4-turbo-preview")
    
    # Embedding Configuration
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dim: int = Field(default=1536)
    
    # Vector Store Configuration
    vector_store_type: Literal["chroma", "faiss"] = Field(default="chroma")
    chroma_persist_dir: str = Field(default="./data/chroma")
    
    # Chunking Configuration
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    
    # Retrieval Configuration
    top_k: int = Field(default=5)
    rerank_enabled: bool = Field(default=True)
    
    # Generation Configuration
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
