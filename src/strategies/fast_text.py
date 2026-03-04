"""
FastTextExtractor — Cheapest Extraction Strategy
=================================================
Handles digital PDFs (pypdf), HTML (stdlib html.parser), plain text, and
Markdown with no external vision or layout engine.

Confidence heuristic
--------------------
``confidence = useful_chars / total_chars``

where *useful_chars* are alphanumerics + basic punctuation.  Low ratios
indicate pages dominated by binary content or extraction artefacts, which
triggers escalation.
"""

from __future__ import annotations

import html.parser
import io
import logging
import re
from pathlib import Path
from typing import Any

from src.models.schemas import (
    DocumentProfile,
    ExtractedDocument,
    OriginType,
)
from src.strategies.base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

_USEFUL_CHAR_RE = re.compile(r"[A-Za-z0-9.,;:!?'\"\-\(\)\[\]\{\}]")


class _HTMLStripper(html.parser.HTMLParser):
    """Minimal HTML parser that collects visible text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags = {"script", "style", "head", "meta", "link"}
        self._current_tag = ""

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self._current_tag = tag.lower()

    def handle_endtag(self, tag: str) -> None:
        self._current_tag = ""

    def handle_data(self, data: str) -> None:
        if self._current_tag not in self._skip_tags:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


class FastTextExtractor(BaseExtractor):
    """
    Strategy 1: Fast, cheap text extraction.

    Best suited for
    ---------------
    * Born-digital PDFs (selectable text)
    * HTML pages
    * Markdown and plain text files
    * DOCX (basic, via byte scanning for paragraph runs)

    Not suitable for
    ----------------
    * Scanned / image-only PDFs → escalate to LayoutExtractor or VisionExtractor
    * Complex multi-column layouts → escalate to LayoutExtractor
    """

    @property
    def name(self) -> str:
        return "fast_text"

    def extract(
        self,
        file_path: str | Path,
        profile: DocumentProfile,
    ) -> ExtractionResult:
        path = Path(file_path)
        raw_bytes = path.read_bytes()
        warnings: list[str] = []

        if profile.origin_type in (OriginType.DIGITAL_PDF,):
            pages, page_texts, warns = self._extract_pdf(raw_bytes)
            warnings.extend(warns)
        elif profile.origin_type == OriginType.HTML:
            page_texts = [self._extract_html(raw_bytes)]
            pages = 1
        elif profile.origin_type == OriginType.DOCX:
            page_texts = [self._extract_docx(raw_bytes)]
            pages = 1
        else:
            # Markdown, plain text, email, unknown — read as UTF-8
            decoded = raw_bytes.decode("utf-8", errors="replace")
            page_texts = [decoded]
            pages = 1

        full_text = "\n\n".join(t for t in page_texts if t)
        confidence = self._compute_confidence(full_text)

        if profile.origin_type == OriginType.SCANNED_PDF:
            warnings.append(
                "FastTextExtractor: scanned PDF detected — confidence may be low; "
                "consider escalating to VisionExtractor."
            )
            confidence = min(confidence, 0.45)

        doc = ExtractedDocument(
            document_id=profile.document_id,
            full_text=full_text,
            full_text_by_page=page_texts,
            page_count=max(pages, 1),
            overall_confidence=confidence,
            warnings=warnings,
        )
        return ExtractionResult(
            document=doc,
            confidence=confidence,
            strategy_name=self.name,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Format-specific extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pdf(raw_bytes: bytes) -> tuple[int, list[str], list[str]]:
        """Extract per-page text from a digital PDF using pypdf."""
        warnings: list[str] = []
        try:
            import pypdf  # type: ignore[import-not-found]

            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            page_texts: list[str] = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    page_texts.append(text.strip())
                except Exception as exc:
                    warnings.append(f"Page {i + 1}: text extraction failed — {exc}")
                    page_texts.append("")
            return len(reader.pages), page_texts, warnings

        except ImportError:
            warnings.append(
                "pypdf not installed; falling back to raw PDF byte scanning."
            )
            raw_str = raw_bytes.decode("latin-1", errors="replace")
            text_blocks = re.findall(r"BT\s*(.*?)\s*ET", raw_str, re.DOTALL)
            text = " ".join(text_blocks)[:8192]
            return 1, [text], warnings

    @staticmethod
    def _extract_html(raw_bytes: bytes) -> str:
        """Strip HTML tags and return visible text."""
        import chardet

        detected = chardet.detect(raw_bytes[:4096])
        enc = detected.get("encoding") or "utf-8"
        html_str = raw_bytes.decode(enc, errors="replace")
        stripper = _HTMLStripper()
        stripper.feed(html_str)
        return stripper.get_text()

    @staticmethod
    def _extract_docx(raw_bytes: bytes) -> str:
        """
        Extract paragraph runs from a DOCX file.

        Uses python-docx when available; falls back to ZIP-based XML scanning.
        """
        try:
            import docx  # type: ignore[import-not-found]

            doc = docx.Document(io.BytesIO(raw_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            pass

        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                try:
                    xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
                    texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, re.DOTALL)
                    return " ".join(texts)
                except KeyError:
                    return ""
        except zipfile.BadZipFile:
            return ""

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(text: str) -> float:
        """Ratio of useful printable characters to total characters."""
        if not text:
            return 0.0
        total = len(text)
        useful = len(_USEFUL_CHAR_RE.findall(text))
        return round(min(useful / total, 1.0), 4)
