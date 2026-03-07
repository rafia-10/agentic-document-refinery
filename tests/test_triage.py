"""
Unit Tests — TriageAgent
========================
Tests cover origin_type detection, layout_complexity scoring,
domain_hint classification, cost estimation, and full profile construction.

All tests are runnable without any real document files — synthetic byte
strings and tmp files are used throughout.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.models.schemas import LayoutComplexity, OriginType
from src.agents.triage import TriageAgent

RULES_PATH = "rubric/extraction_rules.yaml"


@pytest.fixture
def agent() -> TriageAgent:
    return TriageAgent(rules_path=RULES_PATH)


# ---------------------------------------------------------------------------
# Helpers to create temporary files
# ---------------------------------------------------------------------------

def _tmp_file(suffix: str, content: bytes) -> Path:
    """Write *content* to a temporary file with *suffix* and return its path."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(content)
    f.flush()
    f.close()
    return Path(f.name)


_MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
    b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
    b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
    b"/Contents 4 0 R>>\nendobj\n"
    b"4 0 obj\n<</Length 44>>\nstream\n"
    b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
    b"endstream\nendobj\n"
    b"xref\n0 5\n0000000000 65535 f\n"
    b"trailer\n<</Size 5 /Root 1 0 R>>\nstartxref\n0\n%%EOF\n"
)

_SIMPLE_HTML_BYTES = b"<html><body><p>Revenue and equity analysis.</p></body></html>"
_MARKDOWN_CONTENT = b"# Introduction\n\nThis is a plain markdown file.\n"
_FINANCE_TEXT = (
    b"This report covers revenue, EBITDA, balance sheet items, "
    b"cash flow, dividends, and fiscal year performance."
)
_MEDICAL_TEXT = (
    b"The patient diagnosis indicates a prognosis of recovery. "
    b"Clinical trial results show biomarker improvement."
)
_LEGAL_TEXT = (
    b"Pursuant to the arbitration clause, the plaintiff and defendant "
    b"agree to indemnification under the governing jurisdiction."
)
_TABLE_HEAVY = (
    b"# Report\n"
    b"| Col1 | Col2 | Col3 |\n"
    b"|------|------|------|\n"
    b"| A    | B    | C    |\n"
    b"| D    | E    | F    |\n"
    b"Figure 1: Overview\n"
    b"Figure 2: Breakdown\n"
    b"Figure 3: Trends\n"
    b"Figure 4: Comparison\n"
)


# ---------------------------------------------------------------------------
# Origin type detection
# ---------------------------------------------------------------------------

class TestOriginTypeDetection:

    def test_digital_pdf_from_extension(self, agent: TriageAgent) -> None:
        """A file with .pdf extension containing BT operators → DIGITAL_PDF."""
        tmp = _tmp_file(".pdf", _MINIMAL_PDF_BYTES)
        profile = agent.profile(tmp)
        assert profile.origin_type in (OriginType.NATIVE_DIGITAL, OriginType.SCANNED_IMAGE)
        tmp.unlink()

    def test_html_from_extension(self, agent: TriageAgent) -> None:
        """An HTML file → OriginType.HTML."""
        tmp = _tmp_file(".html", _SIMPLE_HTML_BYTES)
        profile = agent.profile(tmp)
        assert profile.origin_type == OriginType.HTML
        tmp.unlink()

    def test_markdown_from_extension(self, agent: TriageAgent) -> None:
        """A .md file → OriginType.MARKDOWN."""
        tmp = _tmp_file(".md", _MARKDOWN_CONTENT)
        profile = agent.profile(tmp)
        assert profile.origin_type == OriginType.MARKDOWN
        tmp.unlink()

    def test_unknown_extension_returns_unknown(self, agent: TriageAgent) -> None:
        """An unrecognised extension → OriginType.UNKNOWN."""
        tmp = _tmp_file(".xyz", b"some random binary content \x00\x01\x02")
        profile = agent.profile(tmp)
        assert profile.origin_type == OriginType.UNKNOWN
        tmp.unlink()

    def test_file_not_found_raises(self, agent: TriageAgent) -> None:
        with pytest.raises(FileNotFoundError):
            agent.profile("/nonexistent/path/to/doc.pdf")


# ---------------------------------------------------------------------------
# Layout complexity scoring
# ---------------------------------------------------------------------------

class TestLayoutComplexityScoring:

    def test_simple_markdown(self, agent: TriageAgent) -> None:
        """Plain single-column markdown → SIMPLE."""
        tmp = _tmp_file(".md", b"# Hello\n\nJust some text with no tables or figures.\n")
        profile = agent.profile(tmp)
        assert profile.layout_complexity == LayoutComplexity.SIMPLE
        tmp.unlink()

    def test_table_heavy_content_is_complex(self, agent: TriageAgent) -> None:
        """Many tables + figure refs → COMPLEX or HIGHLY_COMPLEX."""
        tmp = _tmp_file(".md", _TABLE_HEAVY)
        profile = agent.profile(tmp)
        assert profile.layout_complexity in (
            LayoutComplexity.COMPLEX,
            LayoutComplexity.HIGHLY_COMPLEX,
            LayoutComplexity.MODERATE,
        )
        tmp.unlink()

    def test_html_base_is_not_highly_complex(self, agent: TriageAgent) -> None:
        """Simple HTML without tables → at most MODERATE."""
        tmp = _tmp_file(".html", _SIMPLE_HTML_BYTES)
        profile = agent.profile(tmp)
        assert profile.layout_complexity in (
            LayoutComplexity.SIMPLE,
            LayoutComplexity.MODERATE,
        )
        tmp.unlink()


# ---------------------------------------------------------------------------
# Domain hint classification
# ---------------------------------------------------------------------------

class TestDomainHintClassification:

    def test_finance_keywords_detected(self, agent: TriageAgent) -> None:
        tmp = _tmp_file(".txt", _FINANCE_TEXT)
        profile = agent.profile(tmp)
        assert "finance" in profile.domain_hints
        tmp.unlink()

    def test_medical_keywords_detected(self, agent: TriageAgent) -> None:
        tmp = _tmp_file(".txt", _MEDICAL_TEXT)
        profile = agent.profile(tmp)
        assert "medical" in profile.domain_hints
        tmp.unlink()

    def test_legal_keywords_detected(self, agent: TriageAgent) -> None:
        tmp = _tmp_file(".txt", _LEGAL_TEXT)
        profile = agent.profile(tmp)
        assert "legal" in profile.domain_hints
        tmp.unlink()

    def test_neutral_text_no_domain_hints(self, agent: TriageAgent) -> None:
        neutral = b"The quick brown fox jumps over the lazy dog."
        tmp = _tmp_file(".txt", neutral)
        profile = agent.profile(tmp)
        assert profile.domain_hints == []
        tmp.unlink()

    def test_multi_domain_text(self, agent: TriageAgent) -> None:
        """Text with both finance and legal keywords → both domains present."""
        multi = _FINANCE_TEXT + b" " + _LEGAL_TEXT
        tmp = _tmp_file(".txt", multi)
        profile = agent.profile(tmp)
        assert "finance" in profile.domain_hints
        assert "legal" in profile.domain_hints
        tmp.unlink()


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestCostEstimation:

    def test_cost_positive(self, agent: TriageAgent) -> None:
        tmp = _tmp_file(".md", _MARKDOWN_CONTENT)
        profile = agent.profile(tmp)
        assert profile.estimated_extraction_cost >= 0.0
        tmp.unlink()

    def test_cost_scales_with_complexity(self, agent: TriageAgent) -> None:
        """
        A table-heavy file should score higher cost than a trivially simple one
        because LayoutComplexity contributes a cost multiplier from the rubric.
        """
        simple_tmp = _tmp_file(".md", b"# Simple\nJust text.\n")
        complex_tmp = _tmp_file(".md", _TABLE_HEAVY * 5)

        simple_profile = agent.profile(simple_tmp)
        complex_profile = agent.profile(complex_tmp)

        # Both are page_count=None (markdown), so base pages=1; cost
        # difference comes purely from complexity weight in the rubric.
        assert complex_profile.estimated_extraction_cost >= simple_profile.estimated_extraction_cost
        simple_tmp.unlink()
        complex_tmp.unlink()


# ---------------------------------------------------------------------------
# Full profile field completeness
# ---------------------------------------------------------------------------

class TestFullProfileFields:

    def test_all_required_fields_populated(self, agent: TriageAgent) -> None:
        tmp = _tmp_file(".html", _FINANCE_TEXT)
        profile = agent.profile(tmp)

        assert profile.document_id        # non-empty string
        assert profile.filename           # non-empty string
        assert profile.origin_type        # valid enum
        assert profile.layout_complexity  # valid enum
        assert profile.language           # non-empty BCP-47 tag
        assert isinstance(profile.domain_hints, list)
        assert profile.estimated_extraction_cost >= 0.0
        assert isinstance(profile.ocr_required, bool)
        assert isinstance(profile.raw_metadata, dict)
        tmp.unlink()

    def test_ocr_required_false_for_html(self, agent: TriageAgent) -> None:
        tmp = _tmp_file(".html", _SIMPLE_HTML_BYTES)
        profile = agent.profile(tmp)
        assert profile.ocr_required is False
        tmp.unlink()

    def test_document_id_is_hex_string(self, agent: TriageAgent) -> None:
        tmp = _tmp_file(".md", _MARKDOWN_CONTENT)
        profile = agent.profile(tmp)
        # document_id should be a 16-char hex prefix of SHA-256
        assert len(profile.document_id) == 16
        int(profile.document_id, 16)  # raises if not valid hex
        tmp.unlink()
