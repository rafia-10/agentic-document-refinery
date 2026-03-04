"""
BaseExtractor — Abstract Interface for All Extraction Strategies
================================================================
Every concrete extractor must implement :meth:`extract` which accepts a
file path and the document's :class:`~src.models.DocumentProfile` and
returns a typed :class:`ExtractionResult` containing the normalised
document and a confidence score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from src.models.schemas import DocumentProfile, ExtractedDocument


@dataclass
class ExtractionResult:
    """
    Return value of :meth:`BaseExtractor.extract`.

    Attributes
    ----------
    document:
        The normalised, fully populated :class:`ExtractedDocument`.
    confidence:
        Extraction confidence in [0, 1].  Values below the rubric threshold
        will trigger escalation to the next strategy tier.
    strategy_name:
        Human-readable name of the strategy that produced this result.
    warnings:
        Non-fatal messages collected during extraction.
    """

    document: ExtractedDocument
    confidence: float
    strategy_name: str
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ExtractionResult.confidence must be in [0, 1], got {self.confidence}"
            )


class BaseExtractor(ABC):
    """
    Abstract base class for all extraction strategies.

    Subclasses implement a single method — :meth:`extract` — which does
    its best to extract content from the given file and returns an
    :class:`ExtractionResult` with a self-assessed confidence score.
    The :class:`~src.agents.extractor.ExtractionRouter` uses the
    confidence score to decide whether to escalate.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short strategy identifier used in logging and ledger entries."""
        ...

    @abstractmethod
    def extract(
        self,
        file_path: str | Path,
        profile: DocumentProfile,
    ) -> ExtractionResult:
        """
        Extract content from *file_path* guided by *profile*.

        Parameters
        ----------
        file_path:
            Path to the document file.
        profile:
            Pre-computed :class:`DocumentProfile` from the triage stage.

        Returns
        -------
        ExtractionResult
            Contains the extracted document, confidence score, and any
            non-fatal warnings emitted during extraction.

        Raises
        ------
        NotImplementedError
            Raised by :class:`~src.strategies.vision.VisionExtractor` when
            the optional OCR backend is not installed.
        """
        ...
