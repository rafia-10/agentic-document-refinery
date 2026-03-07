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
                
                # Decompose table rows into individual facts
                self._extract_facts_from_table(cursor, table, doc.document_id)
                
            conn.commit()

    def clear(self) -> None:
        """Wipes all rows from both `facts` and `tables`.

        Useful for tests or resetting the database without dropping the file.
        """
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM facts")
            cursor.execute("DELETE FROM tables")
            conn.commit()

    def drop(self) -> None:
        """Deletes the underlying database file entirely.

        Use with caution; subsequent operations will recreate an empty database.
        """
        try:
            self._db_path.unlink()
        except FileNotFoundError:
            pass

    def _extract_facts_from_table(self, cursor, table: TableData, document_id: str):
        """Extract numerical facts from a table."""
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell in enumerate(row):
                if isinstance(cell, (int, float)) and isinstance(cell, (int, float)):  # Numerical value
                    header = table.headers[col_idx] if col_idx < len(table.headers) else f"col_{col_idx}"
                    cursor.execute(
                        "INSERT INTO facts (document_id, page_number, fact_type, entity, value, context, source_ldu_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (document_id, table.page_references[0] if table.page_references else 1, "numerical", header, float(cell), f"Table {table.table_id} row {row_idx}", table.table_id)
                    )

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

    def get_numerical_facts(self, document_id: str = None, limit: int = 100) -> list[dict]:
        """Retrieve numerical facts, optionally filtered by document."""
        query = "SELECT * FROM facts WHERE fact_type = 'numerical'"
        params = []
        if document_id:
            query += " AND document_id = ?"
            params.append(document_id)
        query += f" LIMIT {limit}"
        
        rows = self.query_facts(query)
        facts = []
        for row in rows:
            fact = {
                "id": row[0],
                "document_id": row[1],
                "page_number": row[2],
                "fact_type": row[3],
                "entity": row[4],
                "value": row[5],
                "unit": row[6],
                "context": row[7],
                "source_ldu_id": row[8]
            }
            facts.append(fact)
        return facts
