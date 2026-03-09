import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.utils.llm import get_embeddings_model
from src.models.schemas import LDU

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Data Layer — FAISS vector store for semantic search over LDUs.
    """

    def __init__(self, index_path: str | Path = ".refinery/vector_store/"):
        self._index_path = Path(index_path)
        self._index_path.mkdir(parents=True, exist_ok=True)
        
        # LLM Setup
        self._embeddings = get_embeddings_model()
        
        # Persistence
        self._vector_db = None
        self._load_vector_db()

    def _load_vector_db(self):
        """Loads the FAISS index from disk if it exists, else initializes a new one."""
        if (self._index_path / "index.faiss").exists():
            try:
                self._vector_db = FAISS.load_local(
                    str(self._index_path), 
                    self._embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded FAISS index from %s", self._index_path)
            except Exception as e:
                logger.error("Failed to load FAISS index: %s", e)
                self._init_new_db()
        else:
            self._init_new_db()

    def _init_new_db(self):
        """Initializes a new FAISS index with dynamic dimensions."""
        # Detect dimension from embeddings model
        test_emb = self._embeddings.embed_query("test")
        dim = len(test_emb)
        logger.info("Initializing FAISS with dimension: %d", dim)
        
        index = faiss.IndexFlatL2(dim)
        self._vector_db = FAISS(
            embedding_function=self._embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )
        logger.info("Initialized new FAISS vector store.")

    def ingest_ldus(self, ldus: list[LDU]):
        """Embeds and adds LDUs to the vector store."""
        if not ldus: return
        logger.info("Ingesting %d LDUs into Vector Store …", len(ldus))
        
        texts = [ldu.content for ldu in ldus]
        metadatas = [ldu.model_dump() for ldu in ldus]
        ids = [ldu.ldu_id for ldu in ldus]
        
        # Add to FAISS
        self._vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        # Save to disk
        self._vector_db.save_local(str(self._index_path))
        logger.info("Saved FAISS index to %s", self._index_path)

    def clear(self) -> None:
        """Remove all entries from the vector store and delete persisted files.

        The next ingestion will recreate an empty index.  This is useful for
        testing or when rebuilding the index from scratch.
        """
        # delete persisted index files
        for fname in ["index.faiss", "index.pkl", "index.json"]:
            path = self._index_path / fname
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        # reset in-memory store
        self._init_new_db()

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Performs semantic search. Handles index dimension mismatch gracefully."""
        if not self._vector_db:
            self._init_new_db()
        try:
            results = self._vector_db.similarity_search_with_score(query, k=k)
        except AssertionError:
            # Dimension mismatch, reinitialize index and return empty
            logger.warning("FAISS index dimension mismatch. Reinitializing index.")
            self._init_new_db()
            return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
        output = []
        for doc, score in results:
            res = doc.metadata
            res["relevance_score"] = float(score)
            output.append(res)
        return output
