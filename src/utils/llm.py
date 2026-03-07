import os
import hashlib
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

# Load environment variables from .env
load_dotenv()

class DeterministicMockEmbeddings(Embeddings):
    """Deterministic hash-based embeddings for credit-limited environments."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]
        
    def embed_query(self, text: str) -> list[float]:
        # Create a deterministic 1536-dim vector from hash
        h = hashlib.sha256(text.encode()).digest()
        # Seed numpy with hash to get deterministic random vector
        np.random.seed(int.from_bytes(h, "big") % 2**32)
        # Ensure it's a flat list of floats
        return np.random.uniform(-1, 1, 1536).tolist()

def get_chat_model(model_name: str = "gpt-4o-mini", temperature: float = 0, purpose: str = "general"):
    """
    Returns a Chat model based on purpose, prioritizing free models if keys are missing.
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # 1. Vision purpose (Requires a vision-capable model)
    if purpose == "vision":
        if openrouter_api_key:
            return ChatOpenAI(
                model="openai/gpt-4o-mini",
                openai_api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature
            )
        # Vision fallback (if allowed)
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    # 2. General/Summary purpose
    if groq_api_key:
        try:
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=groq_api_key,
                temperature=temperature
            )
        except Exception:
            pass

    if openrouter_api_key:
        # Try a few free models that are usually reliable
        return ChatOpenAI(
            model="meta-llama/llama-3.1-8b-instruct:free",
            openai_api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature
        )

    # 3. Final Fallback (if no keys at all, maybe some local mock or just error)
    if (openai_api_key and "sk-" in openai_api_key):
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    
    # Absolute bottom fallback for demo if OPENROUTER_API_KEY is available but not working
    return ChatOpenAI(
        model="openrouter/free",
        openai_api_key=openrouter_api_key if openrouter_api_key else "free-key",
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature
    )

def get_embeddings_model():
    """Returns a local HuggingFace embedding model (all-MiniLM-L6-v2)."""
    try:
        # Use a local free model as requested
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Failed to load HuggingFaceEmbeddings: {e}. Falling back to deterministic mock.")
        return DeterministicMockEmbeddings()
