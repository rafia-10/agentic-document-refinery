import sqlite3
import logging
from pathlib import Path
from typing import Any, List, Optional
from src.models.schemas import ExtractedDocument, TableData

logger = logging.getLogger(__name__)

class FactTableManager:
    """
    Data Layer — SQLite backend for numerical and structured data.
    
    Extracts tabular data from Documents and stores them in a queryable format.
    """

    def __init__(self, db_path: str | Path = ".refinery/fact_table.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Creates the core tables if they don't exist."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            # Table to store raw numerical facts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT,
                    page_number INTEGER,
                    fact_type TEXT,
                    entity TEXT,
                    value REAL,
                    unit TEXT,
                    context TEXT,
                    source_ldu_id TEXT
                )
            ''')
            # Table to store structured tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tables (
                    table_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    headers TEXT, -- JSON array
                    data TEXT     -- JSON array of arrays
                )
            ''')
            conn.commit()

    def ingest_document_facts(self, doc: ExtractedDocument):
        """Extracts facts from tables and text in the document."""
        logger.info("[%s] Ingesting facts into SQLite …", doc.document_id)
        
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Ingest from Tables
            for table in doc.tables:
                import json
                cursor.execute(
                    "INSERT OR REPLACE INTO tables (table_id, document_id, headers, data) VALUES (?, ?, ?, ?)",
                    (table.table_id, doc.document_id, json.dumps(table.headers), json.dumps(table.rows))
                )
                
                # Optionally decompose table rows into individual facts
                # Simplified for this implementation
                
            conn.commit()

    def query_facts(self, sql_query: str) -> list[tuple]:
        """Runs a raw SQL query against the fact table."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                return cursor.fetchall()
        except Exception as e:
            logger.error("SQL query failed: %s", e)
            return []
