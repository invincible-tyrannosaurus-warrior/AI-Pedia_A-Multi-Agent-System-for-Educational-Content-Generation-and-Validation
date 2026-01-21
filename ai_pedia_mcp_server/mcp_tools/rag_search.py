
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from mcp.server.fastmcp import FastMCP

try:
    from config import DATA_DIR
except ImportError:
    # Fallback for independent execution
    import sys
    from pathlib import Path
    # rag_search.py -> mcp_tools -> ai_pedia_mcp_server -> ROOT
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from config import DATA_DIR

logger = logging.getLogger(__name__)

# Constants
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
COLLECTION_NAME = "ai_pedia_knowledge"

class RAGEngine:
    """
    RAG Engine wrapping ChromaDB and SentenceTransformers.
    Singleton-ish usage is recommended to avoid reloading models.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info("Initializing RAGEngine...")
        
        # Ensure directory exists
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Initialize Client (Persistent)
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        
        # 2. Define Embedding Function (using all-MiniLM-L6-v2)
        # Chroma handles the download and caching of the model automatically via sentence_transformers
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 3. Get or Create Collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef
        )
        logger.info(f"RAG Engine Initialized. Collection '{COLLECTION_NAME}' ready.")

    def ingest_document(self, text: str, metadata: dict) -> int:
        """
        Chunk and ingest a document.
        args:
            text: Full text content
            metadata: dict with 'source', 'filename' etc.
        returns:
            Number of chunks added
        """
        if not text:
            return 0
            
        chunks = self._chunk_text(text)
        if not chunks:
            return 0
            
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [metadata for _ in chunks]
        
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"Ingested {len(chunks)} chunks from {metadata.get('filename')}")
        return len(chunks)

    def query(self, query_text: str, n_results: int = 5) -> str:
        """
        Retrieve context for a query.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        documents = results.get("documents", [[]])[0]
        # metadatas = results.get("metadatas", [[]])[0] # Can use if we want search citations
        
        return "\n\n---\n\n".join(documents)

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Simple sliding window chunker."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # If we reached end, break
            if end >= text_len:
                break
                
            # Move start pointer (overlap)
            start += (chunk_size - overlap)
            
        return chunks

# --- MCP Tool Definitions ---

rag_engine = RAGEngine.get_instance()

def rag_query(query: str, n_results: int = 5) -> str:
    """
    Search the local knowledge base (RAG) for relevant context.
    Use this to find definitions, examples, or specific details from uploaded documents.
    """
    try:
        results = rag_engine.query(query, n_results=n_results)
        if not results:
            return "No relevant information found in knowledge base."
        return f"Found relevant context:\n\n{results}"
    except Exception as e:
        return f"RAG Query Failed: {str(e)}"

def register(mcp: FastMCP) -> None:
    mcp.tool()(rag_query)

__all__ = ["rag_query", "register", "RAGEngine"]
