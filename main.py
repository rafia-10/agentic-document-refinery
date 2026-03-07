"""
Document Intelligence Refinery — Advanced CLI Demo
==================================================
Demonstrates Triage, Extraction, PageIndex, and Query with Provenance.

Usage:
  python main.py demo path/to/doc.pdf
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.markup import escape

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import SemanticChunker
from src.agents.indexer import PageIndexer
from src.agents.query_agent import QueryAgent
from src.data.fact_table import FactTableManager
from src.data.vector_store import VectorStoreManager

console = Console()

def run_demo(file_path: str):
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File {file_path} not found.[/red]")
        return

    # Initialize Agents
    triage_agent = TriageAgent()
    router = ExtractionRouter()
    chunker = SemanticChunker()
    indexer = PageIndexer()
    
    # 1. TRIAGE
    console.rule("[bold blue]1. TRIAGE — Document Profiling[/bold blue]")
    with console.status("[bold green]Analyzing document..."):
        profile = triage_agent.profile(path)
    
    # Explain strategy selection
    strategy = "layout" # Default
    if profile.origin_type in ["scanned_image", "mixed"]:
        strategy = "vision"
        reason = "Scanned/Mixed origin detected. Optical character recognition (OCR) required."
    elif profile.origin_type in ["native_digital", "docx", "markdown", "html"]:
        strategy = "fast_text"
        reason = "Clean digital source detected. High-speed text extraction possible."
    else:
        reason = "Standard layout analysis selected for structured extraction."

    profile_table = Table(title=f"DocumentProfile: {profile.document_id}")
    profile_table.add_column("Property", style="cyan")
    profile_table.add_column("Value", style="magenta")
    profile_table.add_row("Filename", profile.filename)
    profile_table.add_row("Origin Type", profile.origin_type)
    profile_table.add_row("Domain Hints", ", ".join(profile.domain_hints) if profile.domain_hints else "N/A")
    profile_table.add_row("Page Count", str(profile.page_count))
    profile_table.add_row("Recommended Strategy", f"[bold yellow]{strategy}[/bold yellow]")
    
    console.print(profile_table)
    console.print(Panel(f"[bold]Strategy Selection Logic:[/bold]\n{escape(reason)}", border_style="yellow"))
    # 2. EXTRACTION
    console.rule("[bold blue]2. EXTRACTION — Multi-Modal Synthesis[/bold blue]")
    with console.status("[bold green]Extracting content..."):
        doc, ledger = router.route(path, profile)
    
    # Side-by-side (Representation)
    snippet_len = 500
    extracted_text = doc.full_text[:snippet_len] + "..."
    
    extraction_panel = Panel(
        f"[bold cyan]Extracted Text (Snippet):[/bold cyan]\n{escape(extracted_text)}",
        title="Extraction Output",
        border_style="green"
    )
    console.print(extraction_panel)

    # Structured JSON Tables
    if doc.tables:
        table_data = doc.tables[0]
        json_table = Table(title=f"Extracted Table: {table_data.table_id}")
        for h in table_data.headers:
            json_table.add_column(h)
        for row in table_data.rows[:5]: # Show first 5 rows
            json_table.add_row(*[str(c) for c in row])
        console.print(json_table)
    else:
        console.print("[italic yellow]No tables detected in this document.[/italic yellow]")

    # Ledger Entry
    ledger_table = Table(title="Extraction Ledger")
    ledger_table.add_column("Strategy", style="cyan")
    ledger_table.add_column("Confidence", style="bold green")
    ledger_table.add_column("Timestamp", style="dim")
    ledger_table.add_row(ledger.final_strategy, f"{ledger.final_confidence:.4f}", ledger.timestamp)
    console.print(ledger_table)

    # 3. PAGE INDEX
    console.rule("[bold blue]3. PAGE INDEX — Hierarchical Navigation[/bold blue]")
    with console.status("[bold green]Building section tree..."):
        ldus = chunker.chunk(doc)
        index = indexer.index(doc, ldus)
    
    tree = Tree(f"Document Index: {escape(index.document_id)}", guide_style="bold bright_blue")
    for sec in index.sections:
        branch = tree.add(f"[bold cyan]{escape(sec.title)}[/bold cyan] (Page {sec.page_references[0]})")
        if sec.summary:
            branch.add(f"[italic]{escape(sec.summary)}[/italic]")
    
    console.print(tree)
    console.print("[dim italic]Navigation enabled: Located information in hierarchical sections without vector search.[/dim italic]")

    # 4. QUERY WITH PROVENANCE
    console.rule("[bold blue]4. QUERY — Advanced RAG with Provenance[/bold blue]")
    
    # Prepare Data Layer
    with console.status("[bold green]Ingesting into Vector Store & Fact Table..."):
        fact_table = FactTableManager()
        fact_table.ingest_document_facts(doc)
        vector_store = VectorStoreManager()
        vector_store.ingest_ldus(ldus)
    
    query_agent = QueryAgent()
    
    default_question = "What is the primary objective of this document?"
    if "finance" in (profile.domain_hints or []):
        default_question = "What was the total revenue or net profit reported?"
    
    console.print(f"[bold]Question:[/bold] {default_question}")
    
    with console.status("[bold green]Agent is thinking..."):
        answer = query_agent.ask(default_question, audit_mode=True)
    
    console.print(Panel(Markdown(answer), title="Agent Answer (with Provenance)", border_style="bold green"))
    
    console.print("\n[bold blue]Demo Complete.[/bold blue]")

def run_triage(file_path: str):
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File {file_path} not found.[/red]")
        return
    triage_agent = TriageAgent()
    console.rule("[bold blue]Stage 1: TRIAGE[/bold blue]")
    with console.status("[bold green]Analyzing..."):
        profile = triage_agent.profile(path)
    
    table = Table(title=f"Document Profile: {profile.document_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Filename", profile.filename)
    table.add_row("Origin Type", profile.origin_type)
    table.add_row("Page Count", str(profile.page_count))
    console.print(table)

def run_extract(file_path: str):
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File {file_path} not found.[/red]")
        return
    triage_agent = TriageAgent()
    router = ExtractionRouter()
    console.rule("[bold blue]Stage 2: EXTRACTION[/bold blue]")
    with console.status("[bold green]Extracting..."):
        profile = triage_agent.profile(path)
        doc, ledger = router.route(path, profile)
    
    console.print(Panel(f"[bold cyan]Raw Content Summary:[/bold cyan]\n{escape(doc.full_text[:500])}...", title="Extraction Result"))
    console.print(f"[dim]Strategy: {ledger.final_strategy} (Confidence: {ledger.final_confidence:.2f})[/dim]")

def run_index(file_path: str):
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File {file_path} not found.[/red]")
        return

    triage_agent = TriageAgent()
    router = ExtractionRouter()
    chunker = SemanticChunker()
    indexer = PageIndexer()

    console.rule("[bold blue]Indexing Document[/bold blue]")
    with console.status("[bold green]Processing..."):
        profile = triage_agent.profile(path)
        doc, _ = router.route(path, profile)
        ldus = chunker.chunk(doc)
        index = indexer.index(doc, ldus)
    
    tree = Tree(f"PageIndex: {escape(index.document_id)}", guide_style="bold bright_blue")
    for sec in index.sections:
        branch = tree.add(f"[bold cyan]{escape(sec.title)}[/bold cyan] (Page {sec.page_references[0]})")
        if sec.summary:
            branch.add(f"[italic]{escape(sec.summary)}[/italic]")
    console.print(tree)

def run_query(question: str):
    console.rule("[bold blue]Query with Provenance[/bold blue]")
    query_agent = QueryAgent()
    console.print(f"[bold]Question:[/bold] {question}")
    with console.status("[bold green]Agent is thinking..."):
        answer = query_agent.ask(question, audit_mode=True)
    console.print(Panel(Markdown(answer), title="Provenance Chain", border_style="bold green"))

def main():
    parser = argparse.ArgumentParser(description="Document Intelligence Refinery CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run the full 4-stage demo")
    demo_parser.add_argument("file", help="Path to document file")

    # Triage command
    triage_parser = subparsers.add_parser("triage", help="Run Stage 1: Triage")
    triage_parser.add_argument("file", help="Path to document file")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Run Stage 2: Extraction")
    extract_parser.add_argument("file", help="Path to document file")

    # Index command
    index_parser = subparsers.add_parser("index", help="Generate and show PageIndex for a document")
    index_parser.add_argument("file", help="Path to document file")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question about the indexed corpus")
    query_parser.add_argument("question", help="The question to ask")
    
    args = parser.parse_args()
    
    if args.command == "demo":
        run_demo(args.file)
    elif args.command == "triage":
        run_triage(args.file)
    elif args.command == "extract":
        run_extract(args.file)
    elif args.command == "index":
        run_index(args.file)
    elif args.command == "query":
        run_query(args.question)

if __name__ == "__main__":
    main()
