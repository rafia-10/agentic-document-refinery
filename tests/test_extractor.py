"""
Unit Tests — ExtractionRouter
==============================
Tests cover strategy selection, confidence-gated escalation,
max-escalation capping, graceful VisionExtractor stub handling,
and ledger entry correctness.

Uses pytest-mock to control strategy confidence without real files.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.extractor import ExtractionRouter, LedgerEntry
from src.models.schemas import (
    DocumentProfile,
    ExtractedDocument,
    LayoutComplexity,
    OriginType,
)
from src.strategies.base import ExtractionResult

RULES_PATH = "rubric/extraction_rules.yaml"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router() -> ExtractionRouter:
    return ExtractionRouter(rules_path=RULES_PATH)


def _make_profile(
    origin: OriginType = OriginType.NATIVE_DIGITAL,
    complexity: LayoutComplexity = LayoutComplexity.SIMPLE,
    page_count: int = 5,
) -> DocumentProfile:
    return DocumentProfile(
        document_id="test-doc-001",
        filename="test.pdf",
        origin_type=origin,
        layout_complexity=complexity,
        page_count=page_count,
        language="en",
        estimated_extraction_cost=10.0,
    )


def _make_extracted_doc(doc_id: str = "test-doc-001") -> ExtractedDocument:
    return ExtractedDocument(
        document_id=doc_id,
        full_text="Sample extracted content.",
        page_count=5,
        overall_confidence=0.9,
    )


def _mock_result(strategy_name: str, confidence: float) -> ExtractionResult:
    return ExtractionResult(
        document=_make_extracted_doc(),
        confidence=confidence,
        strategy_name=strategy_name,
        warnings=[],
    )


def _tmp_file(suffix: str = ".pdf", content: bytes = b"x") -> Path:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

class TestStrategySelection:

    def test_digital_pdf_starts_at_fast_text(self, router: ExtractionRouter) -> None:
        profile = _make_profile(origin=OriginType.NATIVE_DIGITAL)
        assert router._select_initial_strategy(profile) == "fast_text"

    def test_html_starts_at_fast_text(self, router: ExtractionRouter) -> None:
        profile = _make_profile(origin=OriginType.HTML)
        assert router._select_initial_strategy(profile) == "fast_text"

    def test_markdown_starts_at_fast_text(self, router: ExtractionRouter) -> None:
        profile = _make_profile(origin=OriginType.MARKDOWN)
        assert router._select_initial_strategy(profile) == "fast_text"

    def test_scanned_pdf_starts_at_vision(self, router: ExtractionRouter) -> None:
        profile = _make_profile(origin=OriginType.SCANNED_IMAGE)
        assert router._select_initial_strategy(profile) == "vision"

    def test_spreadsheet_starts_at_layout(self, router: ExtractionRouter) -> None:
        profile = _make_profile(origin=OriginType.SPREADSHEET)
        assert router._select_initial_strategy(profile) == "layout"

    def test_unknown_starts_at_layout(self, router: ExtractionRouter) -> None:
        profile = _make_profile(origin=OriginType.UNKNOWN)
        assert router._select_initial_strategy(profile) == "layout"


# ---------------------------------------------------------------------------
# Confidence-gated escalation
# ---------------------------------------------------------------------------

class TestEscalation:

    def test_no_escalation_when_fast_text_passes(self, router: ExtractionRouter) -> None:
        """fast_text confidence ≥ 0.72 → no escalation, ledger shows chain of 1."""
        profile = _make_profile()
        tmp = _tmp_file()

        with patch(
            "src.agents.extractor.FastTextExtractor.extract",
            return_value=_mock_result("fast_text", confidence=0.95),
        ):
            _, ledger = router.route(tmp, profile)

        assert ledger.escalation_count == 0
        assert ledger.final_strategy == "fast_text"
        assert ledger.strategy_chain == ["fast_text"]
        tmp.unlink()

    def test_escalation_on_low_fast_text_confidence(self, router: ExtractionRouter) -> None:
        """fast_text confidence < 0.72 → escalate to layout."""
        profile = _make_profile()
        tmp = _tmp_file()

        with (
            patch(
                "src.agents.extractor.FastTextExtractor.extract",
                return_value=_mock_result("fast_text", confidence=0.50),
            ),
            patch(
                "src.agents.extractor.LayoutExtractor.extract",
                return_value=_mock_result("layout", confidence=0.80),
            ),
        ):
            _, ledger = router.route(tmp, profile)

        assert ledger.escalation_count == 1
        assert ledger.final_strategy == "layout"
        assert "fast_text" in ledger.strategy_chain
        assert "layout" in ledger.strategy_chain
        tmp.unlink()

    def test_double_escalation_to_vision(self, router: ExtractionRouter) -> None:
        """fast_text < 0.72 AND layout < 0.60 → escalate to vision."""
        profile = _make_profile()
        tmp = _tmp_file()

        with (
            patch(
                "src.agents.extractor.FastTextExtractor.extract",
                return_value=_mock_result("fast_text", confidence=0.50),
            ),
            patch(
                "src.agents.extractor.LayoutExtractor.extract",
                return_value=_mock_result("layout", confidence=0.45),
            ),
            patch.dict(os.environ, {"REFINERY_VISION_STUB": "1"}),
        ):
            _, ledger = router.route(tmp, profile)

        # Vision stub returns 0.0 confidence
        assert ledger.escalation_count >= 1
        assert "layout" in ledger.strategy_chain
        tmp.unlink()

    def test_max_escalations_respected(self, router: ExtractionRouter) -> None:
        """Router never escalates more than max_escalations (=2 from rubric)."""
        profile = _make_profile()
        tmp = _tmp_file()

        with (
            patch(
                "src.agents.extractor.FastTextExtractor.extract",
                return_value=_mock_result("fast_text", confidence=0.10),
            ),
            patch(
                "src.agents.extractor.LayoutExtractor.extract",
                return_value=_mock_result("layout", confidence=0.10),
            ),
            patch.dict(os.environ, {"REFINERY_VISION_STUB": "1"}),
        ):
            _, ledger = router.route(tmp, profile)

        assert ledger.escalation_count <= router._max_escalations
        tmp.unlink()


# ---------------------------------------------------------------------------
# VisionExtractor graceful handling
# ---------------------------------------------------------------------------

class TestVisionExtractorGracefulHandling:

    def test_vision_stub_mode_does_not_crash(self, router: ExtractionRouter) -> None:
        """REFINERY_VISION_STUB=1 → router returns a result, no exception."""
        profile = _make_profile(origin=OriginType.SCANNED_IMAGE)
        tmp = _tmp_file()

        with patch.dict(os.environ, {"REFINERY_VISION_STUB": "1"}):
            doc, ledger = router.route(tmp, profile)

        assert doc is not None
        assert ledger is not None
        assert "vision" in ledger.strategy_chain
        tmp.unlink()

    def test_vision_not_implemented_without_stub(self, router: ExtractionRouter) -> None:
        """Without REFINERY_VISION_STUB, VisionExtractor raises but router catches it."""
        profile = _make_profile(origin=OriginType.SCANNED_IMAGE)
        tmp = _tmp_file()

        env = {k: v for k, v in os.environ.items() if k != "REFINERY_VISION_STUB"}
        with patch.dict(os.environ, env, clear=True):
            doc, ledger = router.route(tmp, profile)

        # Router catches NotImplementedError; returns empty doc with warning
        assert doc is not None
        assert any("not available" in w for w in ledger.warnings)
        tmp.unlink()


# ---------------------------------------------------------------------------
# Ledger correctness
# ---------------------------------------------------------------------------

class TestLedgerEntry:

    def test_ledger_is_ledger_entry_instance(self, router: ExtractionRouter) -> None:
        profile = _make_profile()
        tmp = _tmp_file()

        with patch(
            "src.agents.extractor.FastTextExtractor.extract",
            return_value=_mock_result("fast_text", confidence=0.90),
        ):
            _, ledger = router.route(tmp, profile)

        assert isinstance(ledger, LedgerEntry)
        tmp.unlink()

    def test_ledger_to_dict_contains_required_keys(self, router: ExtractionRouter) -> None:
        profile = _make_profile()
        tmp = _tmp_file()

        with patch(
            "src.agents.extractor.FastTextExtractor.extract",
            return_value=_mock_result("fast_text", confidence=0.90),
        ):
            _, ledger = router.route(tmp, profile)

        d = ledger.to_dict()
        required_keys = {
            "document_id", "filename", "strategy_chain", "confidence_scores",
            "cost_estimate", "escalation_count", "final_strategy",
            "final_confidence", "timestamp", "warnings",
        }
        assert required_keys.issubset(d.keys())
        tmp.unlink()

    def test_ledger_document_id_matches_profile(self, router: ExtractionRouter) -> None:
        profile = _make_profile()
        tmp = _tmp_file()

        with patch(
            "src.agents.extractor.FastTextExtractor.extract",
            return_value=_mock_result("fast_text", confidence=0.90),
        ):
            _, ledger = router.route(tmp, profile)

        assert ledger.document_id == profile.document_id
        tmp.unlink()

    def test_ledger_records_confidence_per_strategy(self, router: ExtractionRouter) -> None:
        profile = _make_profile()
        tmp = _tmp_file()

        with (
            patch(
                "src.agents.extractor.FastTextExtractor.extract",
                return_value=_mock_result("fast_text", confidence=0.65),
            ),
            patch(
                "src.agents.extractor.LayoutExtractor.extract",
                return_value=_mock_result("layout", confidence=0.78),
            ),
        ):
            _, ledger = router.route(tmp, profile)

        assert "fast_text" in ledger.confidence_scores
        assert "layout" in ledger.confidence_scores
        assert abs(ledger.confidence_scores["fast_text"] - 0.65) < 0.01
        assert abs(ledger.confidence_scores["layout"] - 0.78) < 0.01
        tmp.unlink()
