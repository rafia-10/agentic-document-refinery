"""
Unit Tests — FactTableManager
==============================
Tests cover SQLite backend for storing and querying tabular data and facts.
"""

import json
import tempfile
from pathlib import Path
import pytest
import sqlite3

from src.data.fact_table import FactTableManager
from src.models.schemas import ExtractedDocument, TableData


@pytest.fixture
def temp_db():
    """Provides a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_fact_table.db"
        yield db_path


@pytest.fixture
def fact_manager(temp_db):
    """Provides a FactTableManager instance with a temporary DB."""
    return FactTableManager(db_path=temp_db)


@pytest.fixture
def sample_extracted_doc():
    """Provides a sample ExtractedDocument with tables."""
    table1 = TableData(
        table_id="table_1",
        page_references=[1],
        headers=["Name", "Age", "Salary"],
        rows=[
            ["Alice", 30, 50000.0],
            ["Bob", 25, 45000.0],
            ["Charlie", 35, 60000.0]
        ]
    )
    table2 = TableData(
        table_id="table_2",
        page_references=[2],
        headers=["Product", "Price", "Quantity"],
        rows=[
            ["Widget A", 10.99, 100],
            ["Widget B", 15.50, 50]
        ]
    )
    doc = ExtractedDocument(
        document_id="test_doc_001",
        full_text="Sample document text.",
        full_text_by_page=["Page 1 text.", "Page 2 text."],
        tables=[table1, table2],
        page_count=2
    )
    return doc


def test_init_creates_tables(fact_manager, temp_db):
    """Test that initialization creates the required tables."""
    # Check if tables exist
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "facts" in tables
        assert "tables" in tables


def test_ingest_document_facts_stores_tables(fact_manager, sample_extracted_doc, temp_db):
    """Test ingesting document facts stores tables correctly."""
    fact_manager.ingest_document_facts(sample_extracted_doc)

    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT table_id, document_id, headers, data FROM tables")
        rows = cursor.fetchall()
        assert len(rows) == 2

        # Check first table
        table_id, doc_id, headers_json, data_json = rows[0]
        assert table_id == "table_1"
        assert doc_id == "test_doc_001"
        headers = json.loads(headers_json)
        data = json.loads(data_json)
        assert headers == ["Name", "Age", "Salary"]
        assert len(data) == 3


def test_query_facts_basic(fact_manager, sample_extracted_doc):
    """Test basic SQL querying of stored facts."""
    fact_manager.ingest_document_facts(sample_extracted_doc)

    # Query all tables
    results = fact_manager.query_facts("SELECT table_id FROM tables")
    assert len(results) == 2
    table_ids = [row[0] for row in results]
    assert "table_1" in table_ids
    assert "table_2" in table_ids


def test_query_facts_invalid_sql(fact_manager):
    """Test handling of invalid SQL queries."""
    results = fact_manager.query_facts("INVALID SQL")
    assert results == []


def test_ingest_empty_document(fact_manager, temp_db):
    """Test ingesting a document with no tables."""
    doc = ExtractedDocument(
        document_id="empty_doc",
        full_text="No tables here.",
        full_text_by_page=["Page text."],
        tables=[],
        page_count=1
    )
    fact_manager.ingest_document_facts(doc)

    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tables")
        count = cursor.fetchone()[0]
        assert count == 0


def test_clear_and_drop_methods(fact_manager, sample_extracted_doc, temp_db):
    """Ensure clear() and drop() behave as advertised."""
    # ingest some data
    fact_manager.ingest_document_facts(sample_extracted_doc)
    assert fact_manager.query_facts("SELECT COUNT(*) FROM tables")[0][0] == 2

    # clear should remove rows but keep file
    fact_manager.clear()
    assert fact_manager.query_facts("SELECT COUNT(*) FROM tables")[0][0] == 0
    assert temp_db.exists()

    # drop should delete the file
    fact_manager.drop()
    assert not temp_db.exists()

    # end of tests