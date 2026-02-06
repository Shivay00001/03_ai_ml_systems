"""
Text Chunkers - Split documents into smaller pieces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from src.config import get_settings


@dataclass
class TextChunk:
    """A chunk of text."""
    
    text: str
    start_idx: int
    end_idx: int
    chunk_idx: int


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[TextChunk]:
        """Split text into chunks."""
        pass


class RecursiveCharacterChunker(BaseChunker):
    """
    Recursive character-based text chunker.
    
    Attempts to split on semantic boundaries (paragraphs, sentences)
    before falling back to character splits.
    """
    
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: Custom separators (ordered by preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.SEPARATORS
    
    def chunk(self, text: str) -> List[TextChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        chunks = self._split_text(text, self.separators)
        
        # Create TextChunk objects with positions
        result = []
        current_idx = 0
        
        for i, chunk_text in enumerate(chunks):
            # Find the actual position in the original text
            start_idx = text.find(chunk_text, current_idx)
            if start_idx == -1:
                start_idx = current_idx
            
            end_idx = start_idx + len(chunk_text)
            
            result.append(
                TextChunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    chunk_idx=i,
                )
            )
            
            # Account for overlap
            current_idx = max(start_idx, end_idx - self.chunk_overlap)
        
        return result
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text."""
        if not separators:
            # Base case: split by character
            return self._split_by_size(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if not separator:
            return self._split_by_size(text)
        
        parts = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # Add separator back (except for last part)
            part_with_sep = part + separator if part != parts[-1] else part
            
            if len(current_chunk) + len(part_with_sep) <= self.chunk_size:
                current_chunk += part_with_sep
            else:
                if current_chunk:
                    if len(current_chunk) > self.chunk_size:
                        # Recursively split with next separator
                        chunks.extend(
                            self._split_text(current_chunk, remaining_separators)
                        )
                    else:
                        chunks.append(current_chunk.strip())
                
                current_chunk = part_with_sep
        
        if current_chunk:
            if len(current_chunk) > self.chunk_size:
                chunks.extend(self._split_text(current_chunk, remaining_separators))
            else:
                chunks.append(current_chunk.strip())
        
        return [c for c in chunks if c.strip()]
    
    def _split_by_size(self, text: str) -> List[str]:
        """Split text by character size."""
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks


class SentenceChunker(BaseChunker):
    """Chunk text by sentences."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 1,  # Number of sentences to overlap
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str) -> List[TextChunk]:
        """Split text into sentence-based chunks."""
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_idx = 0
        
        for sentence in sentences:
            if current_size + len(sentence) > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                start_idx = text.find(chunk_text)
                
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        start_idx=start_idx,
                        end_idx=start_idx + len(chunk_text),
                        chunk_idx=chunk_idx,
                    )
                )
                chunk_idx += 1
                
                # Keep overlap sentences
                current_chunk = current_chunk[-self.chunk_overlap:] if self.chunk_overlap else []
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += len(sentence)
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            start_idx = text.find(chunk_text)
            
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    start_idx=start_idx if start_idx >= 0 else 0,
                    end_idx=len(text),
                    chunk_idx=chunk_idx,
                )
            )
        
        return chunks


def get_chunker(
    chunker_type: str = "recursive",
    **kwargs,
) -> BaseChunker:
    """
    Factory function to get chunker.
    
    Args:
        chunker_type: Type of chunker (recursive, sentence)
        **kwargs: Additional arguments
        
    Returns:
        Chunker instance
    """
    settings = get_settings()
    
    default_kwargs = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    default_kwargs.update(kwargs)
    
    if chunker_type == "recursive":
        return RecursiveCharacterChunker(**default_kwargs)
    elif chunker_type == "sentence":
        return SentenceChunker(**default_kwargs)
    else:
        raise ValueError(f"Unknown chunker type: {chunker_type}")
