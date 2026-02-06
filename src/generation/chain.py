"""
RAG Chain - Complete RAG pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.embeddings import BaseEmbedding, get_embedding_model
from src.core.vector_store import BaseVectorStore, Document, get_vector_store
from src.core.llm import BaseLLM, GenerationConfig, Message, get_llm
from src.ingestion.loaders import BaseLoader, RawDocument, get_loader
from src.ingestion.chunkers import BaseChunker, get_chunker


@dataclass
class RAGResponse:
    """RAG response with sources."""
    
    answer: str
    sources: List[Document]
    usage: Dict[str, int] = field(default_factory=dict)


class RAGChain:
    """
    Complete RAG pipeline.
    
    Handles document ingestion, retrieval, and generation.
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this question."
Be concise and accurate."""
    
    def __init__(
        self,
        embedding_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
        llm: Optional[BaseLLM] = None,
        chunker: Optional[BaseChunker] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize RAG chain.
        
        Args:
            embedding_model: Embedding model
            vector_store: Vector store
            llm: Language model
            chunker: Text chunker
            system_prompt: Custom system prompt
        """
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = vector_store or get_vector_store()
        self.llm = llm or get_llm()
        self.chunker = chunker or get_chunker()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
    
    def ingest_documents(
        self,
        documents: List[RawDocument],
        batch_size: int = 100,
    ) -> int:
        """
        Ingest documents into the vector store.
        
        Args:
            documents: Raw documents to ingest
            batch_size: Batch size for embedding
            
        Returns:
            Number of chunks ingested
        """
        all_chunks = []
        
        for raw_doc in documents:
            # Chunk the document
            chunks = self.chunker.chunk(raw_doc.content)
            
            for chunk in chunks:
                doc = Document(
                    text=chunk.text,
                    metadata={
                        **raw_doc.metadata,
                        "chunk_idx": chunk.chunk_idx,
                        "start_idx": chunk.start_idx,
                        "end_idx": chunk.end_idx,
                    },
                )
                all_chunks.append(doc)
        
        # Embed in batches
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [doc.text for doc in batch]
            embeddings = self.embedding_model.embed_texts(texts)
            
            for doc, embedding in zip(batch, embeddings):
                doc.embedding = embedding
            
            self.vector_store.add_documents(batch)
        
        return len(all_chunks)
    
    def ingest_from_path(self, path: str, **loader_kwargs) -> int:
        """
        Ingest documents from a file or directory.
        
        Args:
            path: File or directory path
            **loader_kwargs: Additional loader arguments
            
        Returns:
            Number of chunks ingested
        """
        loader = get_loader(path, **loader_kwargs)
        documents = loader.load()
        return self.ingest_documents(documents)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        query_embedding = self.embedding_model.embed_text(query)
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        
        return [r.document for r in results]
    
    def generate(
        self,
        query: str,
        context_documents: List[Document],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate response using LLM with context.
        
        Args:
            query: User query
            context_documents: Retrieved documents
            config: Generation configuration
            
        Returns:
            Generated response
        """
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(context_documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[{i}] {doc.text}\nSource: {source}")
        
        context = "\n\n".join(context_parts)
        
        # Build messages
        user_message = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""
        
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_message),
        ]
        
        result = self.llm.generate(messages, config)
        return result.content
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> RAGResponse:
        """
        Complete RAG query: retrieve and generate.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filter
            generation_config: Generation configuration
            
        Returns:
            RAG response with answer and sources
        """
        # Retrieve relevant documents
        documents = self.retrieve(
            query=question,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        
        if not documents:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
            )
        
        # Generate answer
        answer = self.generate(
            query=question,
            context_documents=documents,
            config=generation_config,
        )
        
        return RAGResponse(
            answer=answer,
            sources=documents,
        )
