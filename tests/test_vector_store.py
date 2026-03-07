"""
Unit Tests — VectorStoreManager
===============================
Tests cover FAISS vector store for semantic search over LDUs.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.data.vector_store import VectorStoreManager
from src.models.schemas import LDU, LDUType
from src.utils.llm import DeterministicMockEmbeddings


@pytest.fixture
def temp_index_dir():
    """Provides a temporary directory for FAISS index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_embeddings():
    """Provides a mock embeddings model."""
    return DeterministicMockEmbeddings()


@pytest.fixture
def vector_manager(temp_index_dir, mock_embeddings):
    """Provides a VectorStoreManager with mocked embeddings."""
    with patch('src.data.vector_store.get_embeddings_model', return_value=mock_embeddings):
        manager = VectorStoreManager(index_path=temp_index_dir)
        yield manager


@pytest.fixture
def sample_ldus():
    """Provides sample LDUs for testing."""
    ldu1 = LDU(
        ldu_id="ldu_1",
        document_id="doc_001",
        content="This is a sample paragraph about machine learning.",
        ldu_type=LDUType.PARAGRAPH,
        token_count=10,
        content_hash="hash1",
        page_references=[1],
        sequence_index=0,
    )
    ldu2 = LDU(
        ldu_id="ldu_2",
        document_id="doc_001",
        content="Another section discussing artificial intelligence.",
        ldu_type=LDUType.PARAGRAPH,
        token_count=8,
        content_hash="hash2",
        page_references=[1],
        sequence_index=1,
    )
    ldu3 = LDU(
        ldu_id="ldu_3",
        document_id="doc_002",
        content="Data science involves statistics and programming.",
        ldu_type=LDUType.PARAGRAPH,
        token_count=7,
        content_hash="hash3",
        page_references=[2],
        sequence_index=0,
    )
    return [ldu1, ldu2, ldu3]


def test_init_creates_index_dir(temp_index_dir, mock_embeddings):
    """Test that initialization creates the index directory."""
    with patch('src.data.vector_store.get_embeddings_model', return_value=mock_embeddings):
        manager = VectorStoreManager(index_path=temp_index_dir)
        assert temp_index_dir.exists()
        assert manager._embeddings is mock_embeddings


def test_ingest_ldus_adds_to_store(vector_manager, sample_ldus, temp_index_dir):
    """Test ingesting LDUs adds them to the vector store."""
    vector_manager.ingest_ldus(sample_ldus)

    # Check that index file was created
    assert (temp_index_dir / "index.faiss").exists()
    assert (temp_index_dir / "index.pkl").exists()

    # Check that LDUs are stored
    assert vector_manager._vector_db is not None


def test_search_returns_results(vector_manager, sample_ldus):
    """Test semantic search returns relevant results."""
    vector_manager.ingest_ldus(sample_ldus)

    results = vector_manager.search("machine learning", k=2)
    assert len(results) == 2
    assert all("relevance_score" in result for result in results)
    assert all(isinstance(result["relevance_score"], float) for result in results)


def test_search_empty_store(vector_manager):
    """Test search on empty store returns empty results."""
    results = vector_manager.search("test query")
    assert results == []


def test_persistence_loads_existing_index(temp_index_dir, mock_embeddings, sample_ldus):
    """Test that manager loads existing index from disk."""
    # First, create and save an index
    with patch('src.data.vector_store.get_embeddings_model', return_value=mock_embeddings):
        manager1 = VectorStoreManager(index_path=temp_index_dir)
        manager1.ingest_ldus(sample_ldus)

    # Now create a new manager and check it loads the index
    with patch('src.data.vector_store.get_embeddings_model', return_value=mock_embeddings):
        manager2 = VectorStoreManager(index_path=temp_index_dir)
        # Should load existing index
        results = manager2.search("machine learning", k=1)
        assert len(results) == 1


def test_ingest_empty_list(vector_manager):
    """Test ingesting an empty list of LDUs does nothing."""
    initial_state = vector_manager._vector_db
    vector_manager.ingest_ldus([])
    # Should not change
    assert vector_manager._vector_db is initial_state


def test_search_with_k_larger_than_store(vector_manager, sample_ldus):
    """Test search with k larger than stored items."""
    vector_manager.ingest_ldus(sample_ldus[:2])  # Only 2 LDUs

    results = vector_manager.search("test", k=5)
    assert len(results) == 2  # Should return only available items


def test_clear_resets_store(vector_manager, sample_ldus, temp_index_dir):
    """Confirm that clear() removes all entries and deletes files."""
    vector_manager.ingest_ldus(sample_ldus)
    # index files should exist
    assert (temp_index_dir / "index.faiss").exists()

    vector_manager.clear()
    # after clearing, the directory may exist but files should be removed
    assert not (temp_index_dir / "index.faiss").exists()
    # in-memory vector_db should still be valid but empty
    results = vector_manager.search("anything")
    assert results == []
