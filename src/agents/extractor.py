"""
ExtractionRouter — Confidence-Gated Strategy Dispatcher
=========================================================
The router selects the cheapest adequate extraction strategy for a given
document and escalates to higher-cost strategies when the confidence score
falls below the rubric thresholds.

Algorithm
---------
1. Read DocumentProfile to determine the preferred initial strategy.
2. Execute the strategy → receive ExtractionResult.
3. If confidence < threshold[strategy], escalate to the next tier and repeat.
4. Stop when confidence is acceptable, the escalation ladder is exhausted,
   or max_escalations is reached.
5. Return the best result and a LedgerEntry recording the entire chain.

Escalation ladder (from rubric): fast_text → layout → vision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.models.schemas import (
    DocumentProfile,
    ExtractedDocument,
    OriginType,
)
from src.strategies.base import BaseExtractor, ExtractionResult
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor

logger = logging.getLogger(__name__)


@dataclass
class LedgerEntry:
    """
    Records the complete decision trail for a single document extraction.

    Fields
    ------
    document_id:
        Links back to the DocumentProfile.
    filename:
        Original file name for human readability.
    strategy_chain:
        Ordered list of strategy names tried (e.g. ``["fast_text", "layout"]``).
    confidence_scores:
        Confidence score produced by each strategy (name → score).
    cost_estimate:
        Estimated extraction cost carried forward from the profile.
    escalation_count:
        Number of escalations performed (0 = first strategy was sufficient).
    final_strategy:
        Name of the strategy whose result was ultimately used.
    final_confidence:
        Confidence score of the accepted result.
    timestamp:
        ISO-8601 UTC timestamp of extraction completion.
    warnings:
        All non-fatal warnings accumulated across all strategy attempts.
    """

    document_id: str
    filename: str
    strategy_chain: list[str]
    confidence_scores: dict[str, float]
    cost_estimate: float
    escalation_count: int
    final_strategy: str
    final_confidence: float
    timestamp: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "strategy_chain": self.strategy_chain,
            "confidence_scores": self.confidence_scores,
            "cost_estimate": self.cost_estimate,
            "escalation_count": self.escalation_count,
            "final_strategy": self.final_strategy,
            "final_confidence": self.final_confidence,
            "timestamp": self.timestamp,
            "warnings": self.warnings,
        }


class ExtractionRouter:
    """
    Routes a document through the extraction strategy ladder.

    Parameters
    ----------
    rules_path:
        Path to ``rubric/extraction_rules.yaml``.
    """

    _REGISTRY: dict[str, type[BaseExtractor]] = {
        "fast_text": FastTextExtractor,
        "layout": LayoutExtractor,
        "vision": VisionExtractor,
    }

    def __init__(
        self,
        rules_path: str | Path = "rubric/extraction_rules.yaml",
    ) -> None:
        self._rules = self._load_rules(Path(rules_path))
        self._thresholds: dict[str, float] = self._rules.get("thresholds", {})
        self._ladder: list[str] = self._rules["escalation"]["ladder"]
        self._max_escalations: int = self._rules["escalation"].get("max_escalations", 2)
        self._vision_budget: dict[str, Any] = self._rules.get("vision_budget", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        file_path: str | Path,
        profile: DocumentProfile,
    ) -> tuple[ExtractedDocument, LedgerEntry]:
        """
        Extract content from *file_path* using the optimal strategy.

        Parameters
        ----------
        file_path:
            Path to the raw document file.
        profile:
            Pre-computed DocumentProfile from the triage stage.

        Returns
        -------
        tuple[ExtractedDocument, LedgerEntry]
            The best extracted document and a full audit ledger entry.
        """
        path = Path(file_path)
        initial_strategy = self._select_initial_strategy(profile)
        start_idx = self._ladder.index(initial_strategy)

        strategy_chain: list[str] = []
        confidence_scores: dict[str, float] = {}
        all_warnings: list[str] = []
        best_result: ExtractionResult | None = None
        escalation_count = 0

        for strategy_name in self._ladder[start_idx:]:
            # Budget Check for Vision
            if strategy_name == "vision" and self._is_over_budget(profile):
                msg = f"Vision escalation skipped: document size ({profile.page_count} pages) exceeds budget cap."
                logger.warning("[%s] %s", profile.document_id, msg)
                all_warnings.append(msg)
                break

            extractor = self._REGISTRY[strategy_name]()
            logger.info("[%s] Running %s …", profile.document_id, strategy_name)

            try:
                result = extractor.extract(path, profile)
            except NotImplementedError as exc:
                warn_msg = f"{strategy_name}: not available — {str(exc)[:120]}"
                logger.warning(warn_msg)
                all_warnings.append(warn_msg)
                strategy_chain.append(strategy_name)
                confidence_scores[strategy_name] = 0.0
                break

            strategy_chain.append(strategy_name)
            confidence_scores[strategy_name] = result.confidence
            all_warnings.extend(result.warnings)

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

            threshold = self._thresholds.get(strategy_name, 0.0)
            if result.confidence >= threshold:
                logger.info(
                    "[%s] %s confidence %.3f ≥ threshold %.3f — accepted.",
                    profile.document_id,
                    strategy_name,
                    result.confidence,
                    threshold,
                )
                break
            else:
                logger.info(
                    "[%s] %s confidence %.3f < threshold %.3f — escalating.",
                    profile.document_id,
                    strategy_name,
                    result.confidence,
                    threshold,
                )
                escalation_count += 1
                if escalation_count >= self._max_escalations:
                    logger.warning(
                        "[%s] Max escalations (%d) reached; using best result.",
                        profile.document_id,
                        self._max_escalations,
                    )
                    break

        if best_result is None:
            best_result = self._empty_result(profile)
            all_warnings.append(
                "ExtractionRouter: all strategies failed; returning empty doc."
            )

        ledger = LedgerEntry(
            document_id=profile.document_id,
            filename=profile.filename,
            strategy_chain=strategy_chain,
            confidence_scores=confidence_scores,
            cost_estimate=profile.estimated_extraction_cost,
            escalation_count=escalation_count,
            final_strategy=best_result.strategy_name,
            final_confidence=best_result.confidence,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            warnings=all_warnings,
        )
        return best_result.document, ledger

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_initial_strategy(self, profile: DocumentProfile) -> str:
        """Choose the starting strategy from the ladder based on the profile."""
        if profile.origin_type == OriginType.SCANNED_PDF:
            return "vision"
        if profile.origin_type in (
            OriginType.DIGITAL_PDF,
            OriginType.DOCX,
            OriginType.MARKDOWN,
            OriginType.HTML,
            OriginType.EMAIL,
        ):
            return "fast_text"
        return "layout"

    def _is_over_budget(self, profile: DocumentProfile) -> bool:
        """Check if profile exceeds vision budget constraints defined in rubric."""
        max_pages = self._vision_budget.get("max_pages_per_doc", 10)
        actual_pages = profile.page_count or 1
        return actual_pages > max_pages

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(profile: DocumentProfile) -> ExtractionResult:
        doc = ExtractedDocument(
            document_id=profile.document_id,
            full_text="",
            page_count=profile.page_count or 1,
            overall_confidence=0.0,
        )
        return ExtractionResult(
            document=doc,
            confidence=0.0,
            strategy_name="none",
        )

    @staticmethod
    def _load_rules(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"ExtractionRouter: rubric not found at {path}")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
