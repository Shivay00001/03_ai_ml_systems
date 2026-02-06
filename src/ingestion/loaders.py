"""
Document Loaders - Load documents from various sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RawDocument:
    """Raw document from loader."""
    
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self) -> List[RawDocument]:
        """Load documents from source."""
        pass


class TextFileLoader(BaseLoader):
    """Load plain text files."""
    
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.path = Path(path)
        self.encoding = encoding
    
    def load(self) -> List[RawDocument]:
        """Load text file."""
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        content = self.path.read_text(encoding=self.encoding)
        
        return [
            RawDocument(
                content=content,
                source=str(self.path),
                metadata={
                    "source": str(self.path),
                    "file_type": "text",
                    "file_name": self.path.name,
                },
            )
        ]


class MarkdownLoader(BaseLoader):
    """Load Markdown files."""
    
    def __init__(self, path: str):
        self.path = Path(path)
    
    def load(self) -> List[RawDocument]:
        """Load Markdown file."""
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        content = self.path.read_text(encoding="utf-8")
        
        return [
            RawDocument(
                content=content,
                source=str(self.path),
                metadata={
                    "source": str(self.path),
                    "file_type": "markdown",
                    "file_name": self.path.name,
                },
            )
        ]


class PDFLoader(BaseLoader):
    """Load PDF files."""
    
    def __init__(self, path: str):
        self.path = Path(path)
    
    def load(self) -> List[RawDocument]:
        """Load PDF file."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf required: pip install pypdf")
        
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        reader = PdfReader(self.path)
        documents = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append(
                    RawDocument(
                        content=text,
                        source=str(self.path),
                        metadata={
                            "source": str(self.path),
                            "file_type": "pdf",
                            "file_name": self.path.name,
                            "page": i + 1,
                            "total_pages": len(reader.pages),
                        },
                    )
                )
        
        return documents


class DirectoryLoader(BaseLoader):
    """Load all supported files from a directory."""
    
    SUPPORTED_EXTENSIONS = {
        ".txt": TextFileLoader,
        ".md": MarkdownLoader,
        ".pdf": PDFLoader,
    }
    
    def __init__(
        self,
        path: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ):
        self.path = Path(path)
        self.recursive = recursive
        self.extensions = extensions or list(self.SUPPORTED_EXTENSIONS.keys())
    
    def load(self) -> List[RawDocument]:
        """Load all files from directory."""
        if not self.path.exists():
            raise FileNotFoundError(f"Directory not found: {self.path}")
        
        documents = []
        
        pattern = "**/*" if self.recursive else "*"
        
        for file_path in self.path.glob(pattern):
            if not file_path.is_file():
                continue
            
            ext = file_path.suffix.lower()
            if ext not in self.extensions:
                continue
            
            loader_class = self.SUPPORTED_EXTENSIONS.get(ext)
            if loader_class:
                try:
                    loader = loader_class(str(file_path))
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents


def get_loader(path: str, **kwargs) -> BaseLoader:
    """
    Factory function to get appropriate loader.
    
    Args:
        path: File or directory path
        **kwargs: Additional arguments
        
    Returns:
        Appropriate loader instance
    """
    path_obj = Path(path)
    
    if path_obj.is_dir():
        return DirectoryLoader(path, **kwargs)
    
    ext = path_obj.suffix.lower()
    
    if ext == ".txt":
        return TextFileLoader(path, **kwargs)
    elif ext == ".md":
        return MarkdownLoader(path)
    elif ext == ".pdf":
        return PDFLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
