"""
Triage Agent
============
Inspects a raw document file and produces a :class:`~src.models.DocumentProfile`
describing the document's origin, layout, language, domain, and estimated
extraction cost — all without performing any deep content extraction.

Algorithm
---------
1. Detect origin type via python-magic (MIME-type) with an extension fallback.
2. Score layout complexity using a rule table keyed on origin + simple
   textual heuristics (table-like lines, multi-column markers, figure refs).
3. Classify domain hints using a keyword-bag loaded from the rubric YAML.
4. Detect primary language with chardet (on first 8 KB of decoded text).
5. Estimate extraction cost using the rubric weight table.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

import chardet
import yaml

from src.models.schemas import (
    DocumentProfile,
    LayoutComplexity,
    OriginType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension → OriginType fallback table (used when magic is unavailable)
# ---------------------------------------------------------------------------
_EXT_ORIGIN: dict[str, OriginType] = {
    ".pdf": OriginType.DIGITAL_PDF,  # assumed digital unless magic says otherwise
    ".html": OriginType.HTML,
    ".htm": OriginType.HTML,
    ".md": OriginType.MARKDOWN,
    ".markdown": OriginType.MARKDOWN,
    ".docx": OriginType.DOCX,
    ".doc": OriginType.DOCX,
    ".csv": OriginType.SPREADSHEET,
    ".xlsx": OriginType.SPREADSHEET,
    ".xls": OriginType.SPREADSHEET,
    ".eml": OriginType.EMAIL,
    ".msg": OriginType.EMAIL,
    ".txt": OriginType.MARKDOWN,  # treat plain text like markdown (simple linear)
}

# MIME → OriginType (python-magic produces these)
_MIME_ORIGIN: dict[str, OriginType] = {
    "application/pdf": OriginType.DIGITAL_PDF,
    "text/html": OriginType.HTML,
    "text/markdown": OriginType.MARKDOWN,
    "text/plain": OriginType.MARKDOWN,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": OriginType.DOCX,
    "application/msword": OriginType.DOCX,
    "application/vnd.ms-excel": OriginType.SPREADSHEET,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": OriginType.SPREADSHEET,
    "text/csv": OriginType.SPREADSHEET,
    "message/rfc822": OriginType.EMAIL,
}

# Regex patterns indicating layout complexity
_TABLE_LINE_RE = re.compile(r"(\|.+\||\t.+\t.+)", re.MULTILINE)
_FIGURE_REF_RE = re.compile(r"(figure|fig\.|chart|diagram|image)\s+\d+", re.IGNORECASE)
_EQUATION_RE = re.compile(r"(\$\$?.+?\$\$?|\\begin\{equation\})", re.DOTALL)
_MULTI_COL_RE = re.compile(r"  {4,}", re.MULTILINE)  # 4+ consecutive spaces → column hint


class TriageAgent:
    """
    Entry point for the triage stage.

    Parameters
    ----------
    rules_path:
        Path to ``rubric/extraction_rules.yaml``.  Loaded once at construction.
    """

    def __init__(self, rules_path: str | Path = "rubric/extraction_rules.yaml") -> None:
        self._rules = self._load_rules(Path(rules_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, file_path: str | Path) -> DocumentProfile:
        """
        Inspect *file_path* and return a :class:`DocumentProfile`.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the document to triage.

        Returns
        -------
        DocumentProfile
            Fully populated profile ready for downstream routing.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"TriageAgent: file not found — {path}")

        raw_bytes = path.read_bytes()
        document_id = hashlib.sha256(raw_bytes).hexdigest()[:16]

        origin_type = self._detect_origin_type(path, raw_bytes)
        sample_text = self._extract_text_sample(raw_bytes)
        language = self._detect_language(sample_text)
        layout_complexity = self._score_layout_complexity(origin_type, sample_text)
        domain_hints = self._classify_domain_hints(sample_text)
        page_count = self._estimate_page_count(origin_type, raw_bytes)
        ocr_required = origin_type == OriginType.SCANNED_PDF
        cost = self._estimate_cost(origin_type, layout_complexity, page_count)

        return DocumentProfile(
            document_id=document_id,
            filename=path.name,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            page_count=page_count,
            language=language,
            domain_hints=domain_hints,
            estimated_extraction_cost=cost,
            ocr_required=ocr_required,
            raw_metadata={"file_size_bytes": len(raw_bytes)},
        )

    # ------------------------------------------------------------------
    # Origin detection
    # ------------------------------------------------------------------

    def _detect_origin_type(self, path: Path, raw_bytes: bytes) -> OriginType:
        """
        Try python-magic first; fall back to extension lookup.

        PDF files are further probed for image-only pages (scanned).
        """
        mime = self._magic_mime(raw_bytes)
        if mime:
            origin = _MIME_ORIGIN.get(mime)
            if origin:
                # Refine: PDFs with no embedded text are scanned
                if origin == OriginType.DIGITAL_PDF and self._is_image_pdf(raw_bytes):
                    return OriginType.SCANNED_PDF
                return origin

        # Extension fallback
        ext = path.suffix.lower()
        return _EXT_ORIGIN.get(ext, OriginType.UNKNOWN)

    @staticmethod
    def _magic_mime(raw_bytes: bytes) -> str | None:
        """Return MIME type string or None if python-magic is unavailable."""
        try:
            import magic  # type: ignore[import-not-found]
            return magic.from_buffer(raw_bytes[:4096], mime=True)
        except ImportError:
            logger.debug("python-magic not installed; falling back to extension lookup")
            return None

    @staticmethod
    def _is_image_pdf(raw_bytes: bytes) -> bool:
        """
        Heuristic: a PDF is likely scanned/image-only if it contains very
        few embedded text characters relative to its page markers, OR
        a high density of image objects.
        """
        try:
            # Count Begin Text operators and Page markers
            text_markers = len(re.findall(rb"BT\s", raw_bytes))
            page_markers = len(re.findall(rb"/Page\b", raw_bytes))
            image_markers = len(re.findall(rb"/XObject\s*<</Subtype\s*/Image", raw_bytes))

            if page_markers == 0:
                return False

            # Density of text blocks per page
            text_density = text_markers / page_markers
            # Density of images per page
            image_density = image_markers / page_markers

            # If very low text OR high image-to-text ratio
            return text_density < 1.0 or (image_density > 2.0 and text_density < 5.0)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_sample(raw_bytes: bytes, max_bytes: int = 8192) -> str:
        """Decode the first *max_bytes* of raw content to a Python string."""
        sample = raw_bytes[:max_bytes]
        detected = chardet.detect(sample)
        enc = detected.get("encoding") or "utf-8"
        try:
            return sample.decode(enc, errors="replace")
        except (LookupError, UnicodeDecodeError):
            return sample.decode("utf-8", errors="replace")

    @staticmethod
    def _detect_language(text: str) -> str:
        """
        Return a BCP-47 tag for the dominant language in *text*.

        Uses chardet's Unicode script detection as a lightweight heuristic.
        Falls back to ``"en"`` when detection is inconclusive.
        """
        clean = re.sub(r"<[^>]+>", " ", text)
        detected = chardet.detect(clean.encode("utf-8", errors="replace"))
        lang = detected.get("language") or ""

        _chardet_to_bcp47: dict[str, str] = {
            "English": "en",
            "French": "fr",
            "German": "de",
            "Spanish": "es",
            "Italian": "it",
            "Russian": "ru",
            "Arabic": "ar",
            "Chinese": "zh",
            "Japanese": "ja",
            "Korean": "ko",
        }
        return _chardet_to_bcp47.get(lang, "en")

    # ------------------------------------------------------------------
    # Layout complexity scoring
    # ------------------------------------------------------------------

    def _score_layout_complexity(
        self, origin_type: OriginType, sample_text: str
    ) -> LayoutComplexity:
        """
        Assign a layout complexity tier from a simple rule table.

        Scoring is additive: each detected feature increments a score.
        Thresholds map the score to a :class:`LayoutComplexity` tier.
        """
        score = 0

        _origin_base: dict[OriginType, int] = {
            OriginType.SCANNED_PDF:  3,
            OriginType.DIGITAL_PDF:  2,
            OriginType.DOCX:         1,
            OriginType.SPREADSHEET:  3,
            OriginType.HTML:         1,
            OriginType.MARKDOWN:     0,
            OriginType.EMAIL:        0,
            OriginType.UNKNOWN:      2,
        }
        score += _origin_base.get(origin_type, 1)

        table_matches = len(_TABLE_LINE_RE.findall(sample_text))
        figure_matches = len(_FIGURE_REF_RE.findall(sample_text))
        equation_matches = len(_EQUATION_RE.findall(sample_text))
        multi_col = bool(_MULTI_COL_RE.search(sample_text))

        if table_matches >= 5:
            score += 3
        elif table_matches >= 2:
            score += 1

        if figure_matches >= 3:
            score += 2
        elif figure_matches >= 1:
            score += 1

        if equation_matches >= 2:
            score += 2
        elif equation_matches == 1:
            score += 1

        if multi_col:
            score += 1

        if score <= 2:
            return LayoutComplexity.SIMPLE
        elif score <= 5:
            return LayoutComplexity.MODERATE
        elif score <= 8:
            return LayoutComplexity.COMPLEX
        else:
            return LayoutComplexity.HIGHLY_COMPLEX

    # ------------------------------------------------------------------
    # Domain classification
    # ------------------------------------------------------------------

    def _classify_domain_hints(self, sample_text: str) -> list[str]:
        """
        Return a list of domain labels whose keywords appear in *sample_text*.

        Keyword matching is case-insensitive against each domain's keyword
        list from the rubric YAML.
        """
        domain_keywords: dict[str, list[str]] = self._rules.get("domain_keywords", {})
        lower_text = sample_text.lower()
        matched: list[str] = []
        for domain, keywords in domain_keywords.items():
            for kw in keywords:
                if kw.lower() in lower_text:
                    matched.append(domain)
                    break
        return matched

    # ------------------------------------------------------------------
    # Page count & cost estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_page_count(origin_type: OriginType, raw_bytes: bytes) -> int | None:
        """
        Cheaply estimate page count without full parsing.
        """
        if origin_type in (OriginType.DIGITAL_PDF, OriginType.SCANNED_PDF):
            count = len(re.findall(rb"/Type\s*/Page\b", raw_bytes))
            return max(1, count) if count > 0 else None

        if origin_type == OriginType.DOCX:
            # DOCX is a ZIP. app.xml usually contains <Pages>N</Pages>
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                    app_xml = zf.read("docProps/app.xml").decode("utf-8", errors="ignore")
                    match = re.search(r"<Pages>(\d+)</Pages>", app_xml)
                    if match:
                        return int(match.group(1))
            except Exception:
                pass

        if origin_type in (OriginType.HTML, OriginType.MARKDOWN, OriginType.EMAIL):
            # Estimate pages by line count (roughly 50 lines per page)
            lines = raw_bytes.count(b"\n")
            return max(1, lines // 50 + 1)

        return None

    def _estimate_cost(
        self,
        origin_type: OriginType,
        layout_complexity: LayoutComplexity,
        page_count: int | None,
    ) -> float:
        """
        Compute a normalised cost estimate using the rubric weight table.

        ``cost = page_count × origin_weight × complexity_weight``
        """
        weights: dict[str, Any] = self._rules.get("cost_weights", {})
        origin_w: float = weights.get("origin", {}).get(origin_type.value, 1.0)
        complexity_w: float = weights.get("complexity", {}).get(layout_complexity.value, 1.0)
        pages = page_count if page_count is not None else 1
        return round(pages * origin_w * complexity_w, 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rules(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"TriageAgent: rubric not found at {path}")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
