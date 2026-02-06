"""
LLM Clients - Language model abstraction layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings


@dataclass
class Message:
    """Chat message."""
    
    role: str  # system, user, assistant
    content: str


@dataclass
class GenerationConfig:
    """Generation configuration."""
    
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stop: Optional[List[str]] = None


@dataclass
class GenerationResult:
    """Generation result."""
    
    content: str
    finish_reason: str
    usage: Dict[str, int] = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a response."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM client."""
    
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model name
            api_key: Optional API key
        """
        try:
            from openai import OpenAI
            import tiktoken
        except ImportError:
            raise ImportError("openai and tiktoken packages required")
        
        settings = get_settings()
        self._client = OpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model
        
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def generate(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate response using OpenAI.
        
        Args:
            messages: Chat messages
            config: Generation configuration
            
        Returns:
            Generation result
        """
        config = config or GenerationConfig()
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop,
        )
        
        choice = response.choices[0]
        
        return GenerationResult(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "unknown",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoding.encode(text))


class AnthropicLLM(BaseLLM):
    """Anthropic LLM client."""
    
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Anthropic client.
        
        Args:
            model: Model name
            api_key: Optional API key
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
        
        settings = get_settings()
        self._client = Anthropic(api_key=api_key or settings.anthropic_api_key)
        self._model = model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def generate(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate response using Anthropic."""
        config = config or GenerationConfig()
        
        # Extract system message
        system = ""
        chat_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})
        
        response = self._client.messages.create(
            model=self._model,
            max_tokens=config.max_tokens,
            system=system if system else None,
            messages=chat_messages,
        )
        
        return GenerationResult(
            content=response.content[0].text if response.content else "",
            finish_reason=response.stop_reason or "unknown",
            usage={
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (
                    (response.usage.input_tokens + response.usage.output_tokens)
                    if response.usage
                    else 0
                ),
            },
        )
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return len(text) // 4


class MockLLM(BaseLLM):
    """Mock LLM for testing."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        self._responses = responses or ["This is a mock response."]
        self._call_count = 0
    
    def generate(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Return mock response."""
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        
        return GenerationResult(
            content=response,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4


def get_llm(
    provider: str = "openai",
    **kwargs,
) -> BaseLLM:
    """
    Factory function to get LLM client.
    
    Args:
        provider: LLM provider (openai, anthropic, mock)
        **kwargs: Additional arguments
        
    Returns:
        LLM client instance
    """
    if provider == "openai":
        return OpenAILLM(**kwargs)
    elif provider == "anthropic":
        return AnthropicLLM(**kwargs)
    elif provider == "mock":
        return MockLLM(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
