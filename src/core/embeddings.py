"""
Embedding Models - Text to vector conversion.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import hashlib

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model: OpenAI embedding model name
            api_key: Optional API key (uses env var if not provided)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        
        settings = get_settings()
        self._client = OpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model
        self._dimension = settings.embedding_dim
        
        # Cache for embeddings
        self._cache: dict[str, List[float]] = {}
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        
        embedding = response.data[0].embedding
        self._cache[cache_key] = embedding
        
        return embedding
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Check cache first
        cached_results: dict[int, List[float]] = {}
        texts_to_embed: List[tuple[int, str]] = []
        
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                cached_results[i] = self._cache[cache_key]
            else:
                texts_to_embed.append((i, text))
        
        if texts_to_embed:
            response = self._client.embeddings.create(
                model=self._model,
                input=[t[1] for t in texts_to_embed],
            )
            
            for (idx, text), data in zip(texts_to_embed, response.data):
                cache_key = self._cache_key(text)
                self._cache[cache_key] = data.embedding
                cached_results[idx] = data.embedding
        
        return [cached_results[i] for i in range(len(texts))]


class MockEmbedding(BaseEmbedding):
    """Mock embedding for testing."""
    
    def __init__(self, dimension: int = 1536):
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Generate deterministic mock embedding."""
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self._dimension).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


def get_embedding_model(
    provider: str = "openai",
    **kwargs,
) -> BaseEmbedding:
    """
    Factory function to get embedding model.
    
    Args:
        provider: Embedding provider (openai, mock)
        **kwargs: Additional arguments for the model
        
    Returns:
        Embedding model instance
    """
    if provider == "openai":
        return OpenAIEmbedding(**kwargs)
    elif provider == "mock":
        return MockEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
