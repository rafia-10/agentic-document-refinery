"""
Document Intelligence Refinery — CLI Entrypoint
==============================================
Provides a simple command-line interface to triage and extract documents.

Usage:
  python main.py triage path/to/doc.pdf
  python main.py extract path/to/doc.pdf [--output-dir .out]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)

def run_triage(args: argparse.Namespace) -> None:
    agent = TriageAgent()
    profile = agent.profile(args.file)
    print(profile.model_dump_json(indent=2))

def run_extract(args: argparse.Namespace) -> None:
    agent = TriageAgent()
    profile = agent.profile(args.file)
    
    router = ExtractionRouter()
    doc, ledger = router.route(args.file, profile)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save extracted document
    doc_path = output_dir / f"{profile.document_id}_extracted.json"
    doc_path.write_text(doc.model_dump_json(indent=2))
    
    # Save ledger entry
    ledger_path = output_dir / f"{profile.document_id}_ledger.json"
    ledger_path.write_text(json.dumps(ledger.to_dict(), indent=2))
    
    # Save raw text
    text_path = output_dir / f"{profile.document_id}.txt"
    text_path.write_text(doc.full_text)
    
    logging.info("Extraction complete.")
    logging.info("  Document: %s", doc_path)
    logging.info("  Ledger:   %s", ledger_path)
    logging.info("  Text:     %s", text_path)
    
    # Summary to stdout
    print(json.dumps({
        "document_id": profile.document_id,
        "final_strategy": ledger.final_strategy,
        "confidence": ledger.final_confidence,
        "pages": doc.page_count
    }, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser(description="Document Intelligence Refinery CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Triage command
    t_parser = subparsers.add_parser("triage", help="Analyze document and produce a profile")
    t_parser.add_argument("file", help="Path to document file")
    
    # Extract command
    e_parser = subparsers.add_parser("extract", help="Run full extraction pipeline")
    e_parser.add_argument("file", help="Path to document file")
    e_parser.add_argument("--output-dir", default=".refinery/output", help="Directory for output artifacts")
    
    args = parser.parse_args()
    
    try:
        if args.command == "triage":
            run_triage(args)
        elif args.command == "extract":
            run_extract(args)
    except Exception as exc:
        logging.error("Command failed: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    main()
