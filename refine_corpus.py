import logging
import os
import sys
import json
from pathlib import Path
from typing import List

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import SemanticChunker
from src.agents.indexer import PageIndexer
from src.data.fact_table import FactTableManager
from src.data.vector_store import VectorStoreManager
from src.agents.query_agent import QueryAgent

# --- Configuration ---
CORPUS_DIR = Path("data/data")
OUTPUT_DIR = Path(".refinery")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Select 12 documents across 4 classes
SELECTED_DOCS = {
    "Audited Financials": [
        "Audit Report - 2023.pdf",
        "2022_Audited_Financial_Statement_Report.pdf",
        "2021_Audited_Financial_Statement_Report.pdf"
    ],
    "Annual Reports": [
        "CBE ANNUAL REPORT 2023-24.pdf",
        "EthSwitch-10th-Annual-Report-202324.pdf",
        "ETS-Annual-Report-2023_2024.pdf"
    ],
    "CPI Reports": [
        "Consumer Price Index June 2025.pdf",
        "Consumer Price Index July 2025.pdf",
        "Consumer Price Index August 2025.pdf"
    ],
    "Technical/Other": [
        "fta_performance_survey_final_report_2022.pdf",
        "tax_expenditure_ethiopia_2021_22.pdf",
        "Security_Vulnerability_Disclosure_Standard_Procedure_1.pdf"
    ]
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "refine_corpus.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RefineCorpus")

# --- Pipeline Runner ---

class CorpusRefiner:
    def __init__(self):
        self.triage = TriageAgent()
        self.router = ExtractionRouter()
        self.chunker = SemanticChunker()
        self.indexer = PageIndexer()
        self.fact_table = FactTableManager()
        self.vector_store = VectorStoreManager()
        self.query_agent = QueryAgent()

    def process_document(self, file_path: Path):
        logger.info("--- Processing: %s ---", file_path.name)
        
        # 1. Triage
        profile = self.triage.profile(file_path)
        
        # 2. Extract
        doc, ledger = self.router.route(file_path, profile)
        
        # 3. Semantic Chunking (with 5 rules)
        ldus = self.chunker.chunk(doc)
        
        # 4. Hierarchical Indexing (with LLM summaries)
        index = self.indexer.index(doc, ldus)
        
        # 5. Data Layer Ingestion
        self.fact_table.ingest_document_facts(doc)
        self.vector_store.ingest_ldus(ldus)
        
        logger.info("Successfully refined %s", profile.document_id)
        return profile.document_id

    def generate_qa_artifacts(self):
        """Generates 3 Q&A pairs for each of the 4 classes."""
        qa_results = []
        for class_name, _ in SELECTED_DOCS.items():
            logger.info("Generating Q&A for class: %s", class_name)
            # Example queries for each class
            queries = []
            if class_name == "Audited Financials":
                queries = [
                    "What was the net profit for 2021 according to the audited report?",
                    "Identify the total assets of the organization as of December 31, 2022.",
                    "What are the main auditor's opinions mentioned across the 2020-2022 reports?"
                ]
            elif class_name == "Annual Reports":
                queries = [
                    "Summarize the CBE's performance in the 2023-24 financial year.",
                    "What are the core strategic goals discussed in the EthSwitch 10th Annual Report?",
                    "What was the total revenue recorded in the June 2018 Annual Report?"
                ]
            elif class_name == "CPI Reports":
                queries = [
                    "What was the inflation rate reported in the August 2025 CPI report?",
                    "Compare the price indices between July 2025 and August 2025.",
                    "Which category showed the highest price increase in June 2025?"
                ]
            elif class_name == "Technical/Other":
                queries = [
                    "What are the reporting timelines defined in the SVD Standard Procedure?",
                    "Summarize the tax expenditure findings for Ethiopia in 2021-22.",
                    "What pharmaceutical manufacturing opportunities were identified in the 2019 Ethiopia report?"
                ]
            
            for q in queries:
                ans = self.query_agent.ask(q, audit_mode=True)
                qa_results.append({
                    "class": class_name,
                    "question": q,
                    "answer": ans
                })
        
        # Save Q&A artifacts
        qa_path = OUTPUT_DIR / "qa_examples.json"
        with open(qa_path, "w") as f:
            json.dump(qa_results, f, indent=2)
        logger.info("Q&A Artifacts saved to %s", qa_path)

def main():
    refiner = CorpusRefiner()
    doc_ids = []
    
    # Flatten selected docs
    all_files = []
    for docs in SELECTED_DOCS.values():
        all_files.extend(docs)
    
    # Process each document
    for filename in all_files:
        path = CORPUS_DIR / filename
        if path.exists():
            try:
                # Process only a few for safety if we are in a tight loop, 
                # but I will attempt all 12 as requested.
                doc_ids.append(refiner.process_document(path))
            except Exception as e:
                import traceback
                logger.error("Failed to process %s: %s\n%s", filename, e, traceback.format_exc())
                return None
        else:
            logger.warning("File not found: %s", path)
            
    # Generate Q&A
    refiner.generate_qa_artifacts()
    
    logger.info("Corpus Refinement Complete. Processed %d documents.", len(doc_ids))

if __name__ == "__main__":
    main()
