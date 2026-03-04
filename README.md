# Document Intelligence Refinery

A modular, rubric-driven pipeline for extracting, structuring, and indexing heterogeneous documents at production scale.

---

## Architecture at a Glance

```
Raw File → TriageAgent → ExtractionRouter → Chunker → PageIndex → QueryAgent
                              ↑ confidence-gated escalation ↓
                         fast_text → layout → vision (OCR)
```

- **TriageAgent** — inspects raw bytes to produce a `DocumentProfile` (origin, layout complexity, domain, cost estimate) before any expensive extraction begins.
- **ExtractionRouter** — selects the cheapest adequate extraction strategy and escalates based on confidence thresholds from `rubric/extraction_rules.yaml`.
- **Strategies** — `FastTextExtractor` (digital PDF / HTML / DOCX), `LayoutExtractor` (tables, bounding boxes), `VisionExtractor` (OCR stub — see below).
- **Rubric** — all thresholds, cost weights, and domain keywords live in `rubric/extraction_rules.yaml`. No code change needed to tune the pipeline.

---

## Project Structure

```
document-refinery/
├── src/
│   ├── models/
│   │   └── schemas.py          # Pydantic schemas: DocumentProfile, ExtractedDocument, LDU, PageIndex, ProvenanceChain
│   ├── agents/
│   │   ├── triage.py           # TriageAgent
│   │   └── extractor.py        # ExtractionRouter + LedgerEntry
│   └── strategies/
│       ├── base.py             # BaseExtractor ABC + ExtractionResult
│       ├── fast_text.py        # FastTextExtractor
│       ├── layout.py           # LayoutExtractor
│       └── vision.py           # VisionExtractor (stub)
├── rubric/
│   └── extraction_rules.yaml   # Confidence thresholds, cost weights, domain keywords
├── .refinery/
│   ├── profiles/               # 12 DocumentProfile JSON fixtures (3 per origin class)
│   └── extraction_ledger.jsonl # Extraction audit ledger
├── tests/
│   ├── test_triage.py          # TriageAgent unit tests (18 cases)
│   └── test_extractor.py       # ExtractionRouter unit tests (16 cases)
├── DOMAIN_NOTES.md             # Engineering brief: decision tree, failure modes, cost analysis
├── pyproject.toml
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- (Optional) Tesseract OCR — only for `VisionExtractor`

### Install

```bash
git clone <repo-url>
cd document-refinery

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install runtime + dev dependencies
pip install -e ".[dev]"
```

### Optional: OCR support (VisionExtractor)

```bash
sudo apt-get install tesseract-ocr
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
pip install -e ".[ocr]"
```

---

## Usage

### Triage a document

```python
from src.agents.triage import TriageAgent

agent = TriageAgent(rules_path="rubric/extraction_rules.yaml")
profile = agent.profile("path/to/document.pdf")
print(profile.model_dump_json(indent=2))
```

### Extract a document

```python
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter

profile = TriageAgent().profile("path/to/document.pdf")
router = ExtractionRouter()
doc, ledger = router.route("path/to/document.pdf", profile)

print(f"Strategy chain: {ledger.strategy_chain}")
print(f"Final confidence: {ledger.final_confidence:.2%}")
print(doc.full_text[:500])
```

### CI mode (no OCR backend)

Set `REFINERY_VISION_STUB=1` to suppress `NotImplementedError` from `VisionExtractor` and receive a zero-confidence placeholder instead:

```bash
REFINERY_VISION_STUB=1 pytest tests/
```

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v --tb=short
```

Expected output: **34 tests passed**.

Coverage report:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Tuning the Rubric

All extraction parameters live in `rubric/extraction_rules.yaml`:

| Section | What it controls |
|---|---|
| `thresholds` | Confidence floors per strategy before escalation |
| `escalation.ladder` | Order of strategy promotion |
| `escalation.max_escalations` | Hard cap on escalation steps |
| `chunking` | Token budget, overlap, split preference per LDU |
| `cost_weights` | Per-page cost multipliers by origin × complexity |
| `domain_keywords` | Keyword lists for domain classification |

Changes take effect immediately — no code deploy required.

---

## Corpus Artifacts

Pre-generated fixtures live in `.refinery/`:

| Path | Contents |
|---|---|
| `.refinery/profiles/*.json` | 12 `DocumentProfile` snapshots (3× digital_pdf, scanned_pdf, html, docx) |
| `.refinery/extraction_ledger.jsonl` | 12 ledger entries with strategy chains, confidence scores, and warnings |

---

## Engineering Notes

See [`DOMAIN_NOTES.md`](./DOMAIN_NOTES.md) for:
- The exact decision tree distinguishing native-digital vs. scanned-image PDFs
- Failure mode analysis across 4 enterprise document classes
- Cost-quality tradeoff analysis and budget guard philosophy
- Full Mermaid pipeline diagram with confidence-based feedback loop

---

## Dependencies

| Package | Purpose |
|---|---|
| `pydantic>=2.6` | Data model validation |
| `pyyaml>=6.0` | Rubric configuration |
| `chardet>=5.2` | Language + encoding detection |
| `python-magic>=0.4.27` | MIME-type detection |
| `pypdf>=4.1` | PDF text extraction (FastTextExtractor) |
| `pytesseract` *(optional)* | OCR (VisionExtractor, `[ocr]` extra) |
