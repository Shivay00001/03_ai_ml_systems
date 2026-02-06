"""
Vector Store - Storage and retrieval of embeddings.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from src.config import get_settings


@dataclass
class Document:
    """Document with text and metadata."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Search result with score."""
    
    document: Document
    score: float


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, document_ids: List[str]) -> None:
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize ChromaDB store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Optional persistence directory
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError("chromadb package required: pip install chromadb")
        
        settings = get_settings()
        persist_dir = persist_directory or settings.chroma_persist_dir
        
        self._client = chromadb.Client(
            ChromaSettings(
                persist_directory=persist_dir,
                anonymized_telemetry=False,
            )
        )
        
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to ChromaDB.
        
        Args:
            documents: List of documents with embeddings
        """
        if not documents:
            return
        
        self._collection.add(
            ids=[doc.id for doc in documents],
            embeddings=[doc.embedding for doc in documents if doc.embedding],
            documents=[doc.text for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )
        
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = Document(
                    id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                # Convert distance to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Cosine similarity
                
                search_results.append(SearchResult(document=doc, score=score))
        
        return search_results
    
    def delete(self, document_ids: List[str]) -> None:
        """Delete documents by ID."""
        self._collection.delete(ids=document_ids)
    
    def clear(self) -> None:
        """Clear all documents."""
        # Delete and recreate collection
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store for testing."""
    
    def __init__(self):
        self._documents: Dict[str, Document] = {}
        self._embeddings: np.ndarray = np.array([])
        self._ids: List[str] = []
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to memory."""
        for doc in documents:
            self._documents[doc.id] = doc
            self._ids.append(doc.id)
        
        new_embeddings = np.array([doc.embedding for doc in documents if doc.embedding])
        
        if self._embeddings.size == 0:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search using cosine similarity."""
        if self._embeddings.size == 0:
            return []
        
        query = np.array(query_embedding)
        
        # Compute cosine similarity
        similarities = np.dot(self._embeddings, query) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = self._ids[idx]
            doc = self._documents[doc_id]
            
            # Apply metadata filter
            if filter_metadata:
                if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            results.append(SearchResult(document=doc, score=float(similarities[idx])))
        
        return results
    
    def delete(self, document_ids: List[str]) -> None:
        """Delete documents by ID."""
        for doc_id in document_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                idx = self._ids.index(doc_id)
                self._ids.remove(doc_id)
                self._embeddings = np.delete(self._embeddings, idx, axis=0)
    
    def clear(self) -> None:
        """Clear all documents."""
        self._documents = {}
        self._embeddings = np.array([])
        self._ids = []


def get_vector_store(
    store_type: str = "chroma",
    **kwargs,
) -> BaseVectorStore:
    """
    Factory function to get vector store.
    
    Args:
        store_type: Vector store type (chroma, memory)
        **kwargs: Additional arguments
        
    Returns:
        Vector store instance
    """
    if store_type == "chroma":
        return ChromaVectorStore(**kwargs)
    elif store_type == "memory":
        return InMemoryVectorStore()
    else:
        raise ValueError(f"Unknown store type: {store_type}")
