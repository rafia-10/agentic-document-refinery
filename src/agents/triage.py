"""
Triage Agent — Pluggable Classification
======================================
Inspects a raw document file and produces a :class:`~src.models.DocumentProfile`.
Supports a registry of Domain Classifier plugins.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

import chardet
import yaml
from pypdf import PdfReader

from src.models.schemas import (
    DocumentProfile,
    LayoutComplexity,
    OriginType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocols & Base Classes for Pluggability
# ---------------------------------------------------------------------------

class DomainClassifier(Protocol):
    """Interface for domain classification plugins."""
    def classify(self, sample_text: str, rules: dict[str, Any]) -> list[str]:
        ...

class KeywordDomainClassifier:
    """Default keyword-based classifier using rubric YAML."""
    def classify(self, sample_text: str, rules: dict[str, Any]) -> list[str]:
        domain_keywords: dict[str, list[str]] = rules.get("domain_keywords", {})
        lower_text = sample_text.lower()
        matched: list[str] = []
        for domain, keywords in domain_keywords.items():
            for kw in keywords:
                if kw.lower() in lower_text:
                    matched.append(domain)
                    break
        return matched

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXT_ORIGIN: dict[str, OriginType] = {
    ".pdf": OriginType.NATIVE_DIGITAL,
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
    ".txt": OriginType.MARKDOWN,
}

_MIME_ORIGIN: dict[str, OriginType] = {
    "application/pdf": OriginType.NATIVE_DIGITAL,
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

_TABLE_LINE_RE = re.compile(r"(\|.+\||\t.+\t.+)", re.MULTILINE)
_FIGURE_REF_RE = re.compile(r"(figure|fig\.|chart|diagram|image)\s+\d+", re.IGNORECASE)
_EQUATION_RE = re.compile(r"(\$\$?.+?\$\$?|\\begin\{equation\})", re.DOTALL)
_MULTI_COL_RE = re.compile(r"  {4,}", re.MULTILINE)


class TriageAgent:
    """
    Entry point for the triage stage.
    """

    def __init__(self, rules_path: str | Path = "rubric/extraction_rules.yaml") -> None:
        self._rules = self._load_rules(Path(rules_path))
        self._classifiers: list[DomainClassifier] = [KeywordDomainClassifier()]

    def register_classifier(self, classifier: DomainClassifier) -> None:
        """Register a new domain classification plugin."""
        self._classifiers.append(classifier)

    def profile(self, file_path: str | Path) -> DocumentProfile:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"TriageAgent: file not found — {path}")

        raw_bytes = path.read_bytes()
        document_id = hashlib.sha256(raw_bytes).hexdigest()[:16]

        origin_type = self._detect_origin_type(path, raw_bytes)
        sample_text = self._extract_text_sample(raw_bytes)
        language = self._detect_language(sample_text)
        layout_complexity = self._score_layout_complexity(origin_type, sample_text)
        domain_hints = self._run_classification(sample_text)
        page_count = self._estimate_page_count(origin_type, raw_bytes)
        ocr_required = origin_type == OriginType.SCANNED_IMAGE
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

    def _run_classification(self, sample_text: str) -> list[str]:
        """Execute all registered classifiers and return unique hits."""
        all_hints: set[str] = set()
        for classifier in self._classifiers:
            try:
                hints = classifier.classify(sample_text, self._rules)
                all_hints.update(hints)
            except Exception as e:
                logger.error("Classifier %s failed: %s", type(classifier).__name__, e)
        return sorted(list(all_hints))

    def _detect_origin_type(self, path: Path, raw_bytes: bytes) -> OriginType:
        mime = self._magic_mime(raw_bytes)
        ext = path.suffix.lower()
        # Terminology alignment
        origin = OriginType.UNKNOWN
        if mime == "application/pdf":
            # PDF analysis for digital vs scanned vs mixed
            is_scanned, has_digital, digital_ratio = self._analyze_pdf_content(path)
            if has_digital and is_scanned:
                origin = OriginType.MIXED
            elif has_digital:
                origin = OriginType.NATIVE_DIGITAL
            else:
                origin = OriginType.SCANNED_IMAGE
        elif mime == "text/html" or ext == ".html" or ext == ".htm":
            origin = OriginType.HTML
        elif "wordprocessingml" in (mime or "") or "msword" in (mime or "") or ext == ".docx" or ext == ".doc":
            origin = OriginType.DOCX
        elif mime == "text/markdown" or ext == ".md" or ext == ".markdown":
            origin = OriginType.MARKDOWN
        elif "spreadsheet" in (mime or "") or ext in [".csv", ".xlsx", ".xls"]:
            origin = OriginType.SPREADSHEET
        elif "message/rfc822" in (mime or "") or ext == ".eml" or ext == ".msg":
            origin = OriginType.EMAIL
        elif mime == "text/plain" or ext == ".txt":
            origin = OriginType.MARKDOWN # Treat plain text as markdown for now
        
        return origin

    @staticmethod
    def _magic_mime(raw_bytes: bytes) -> str | None:
        try:
            import magic  # type: ignore[import-not-found]
            return magic.from_buffer(raw_bytes[:4096], mime=True)
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"magic failed: {e}")
            return None

    def _analyze_pdf_content(self, path: Path) -> tuple[bool, bool, float]:
        """
        Deeper inspection of PDF to detect digital vs scanned vs mixed.
        Returns (has_scanned_pages, has_digital_pages, digital_page_ratio).
        """
        try:
            reader = PdfReader(path)
            total_pages = len(reader.pages)
            if total_pages == 0:
                return False, False, 0.0
            
            digital_pages = 0
            for page in reader.pages:
                text = page.extract_text() or ""
                if len(text.strip()) > 50: # Threshold for 'digital' content
                    digital_pages += 1
            
            has_digital = digital_pages > 0
            is_scanned = digital_pages < total_pages
            
            return is_scanned, has_digital, (digital_pages / total_pages)
        except Exception as e:
            logger.warning(f"pypdf analysis failed for {path.name}: {e}. Falling back to heuristic.")
            try:
                raw_bytes = path.read_bytes()
                text_markers = len(re.findall(rb"BT\s", raw_bytes))
                page_markers = len(re.findall(rb"/Page\b", raw_bytes)) or 1
                image_markers = len(re.findall(rb"/XObject\s*<</Subtype\s*/Image", raw_bytes))
                
                text_density = text_markers / page_markers
                image_density = image_markers / page_markers
                
                # Heuristic mapping to (is_scanned, has_digital, digital_ratio)
                has_digital = text_density > 0.1
                is_scanned = text_density < 5.0 or image_density > 0.5
                digital_ratio = min(text_density / 10.0, 1.0)
                
                return is_scanned, has_digital, digital_ratio
            except Exception:
                return False, False, 0.0

    @staticmethod
    def _extract_text_sample(raw_bytes: bytes, max_bytes: int = 8192) -> str:
        sample = raw_bytes[:max_bytes]
        detected = chardet.detect(sample)
        enc = detected.get("encoding") or "utf-8"
        try:
            return sample.decode(enc, errors="replace")
        except:
            return sample.decode("utf-8", errors="replace")

    @staticmethod
    def _detect_language(text: str) -> str:
        clean = re.sub(r"<[^>]+>", " ", text)
        detected = chardet.detect(clean.encode("utf-8", errors="replace"))
        lang = detected.get("language") or ""
        _conv = {"English": "en", "French": "fr", "German": "de", "Spanish": "es", 
                 "Italian": "it", "Russian": "ru", "Arabic": "ar", "Chinese": "zh", 
                 "Japanese": "ja", "Korean": "ko"}
        return _conv.get(lang, "en")

    def _score_layout_complexity(self, origin_type: OriginType, text: str) -> LayoutComplexity:
        score = 0
        _base = {OriginType.SCANNED_IMAGE: 3, OriginType.NATIVE_DIGITAL: 2, OriginType.DOCX: 1, 
                 OriginType.SPREADSHEET: 3, OriginType.HTML: 1, OriginType.MARKDOWN: 0, 
                 OriginType.EMAIL: 0, OriginType.UNKNOWN: 2}
        score += _base.get(origin_type, 1)

        t_matches = len(_TABLE_LINE_RE.findall(text))
        f_matches = len(_FIGURE_REF_RE.findall(text))
        e_matches = len(_EQUATION_RE.findall(text))
        multi_col = bool(_MULTI_COL_RE.search(text))

        if t_matches >= 5: score += 3
        elif t_matches >= 2: score += 1
        if f_matches >= 3: score += 2
        elif f_matches >= 1: score += 1
        if e_matches >= 2: score += 2
        elif e_matches == 1: score += 1
        if multi_col: score += 1

        if score <= 2: return LayoutComplexity.SIMPLE
        elif score <= 5: return LayoutComplexity.MODERATE
        elif score <= 8: return LayoutComplexity.COMPLEX
        else: return LayoutComplexity.HIGHLY_COMPLEX

    @staticmethod
    def _estimate_page_count(origin_type: OriginType, raw_bytes: bytes) -> int | None:
        if origin_type in (OriginType.NATIVE_DIGITAL, OriginType.SCANNED_IMAGE, OriginType.MIXED):
            count = len(re.findall(rb"/Type\s*/Page\b", raw_bytes))
            return max(1, count) if count > 0 else None
        if origin_type == OriginType.DOCX:
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                    xml = zf.read("docProps/app.xml").decode("utf-8", errors="ignore")
                    m = re.search(r"<Pages>(\d+)</Pages>", xml)
                    if m: return int(m.group(1))
            except: pass
        if origin_type in (OriginType.HTML, OriginType.MARKDOWN, OriginType.EMAIL):
            return max(1, raw_bytes.count(b"\n") // 50 + 1)
        return None

    def _estimate_cost(self, o: OriginType, l: LayoutComplexity, p: int | None) -> float:
        w = self._rules.get("cost_weights", {})
        o_w = w.get("origin", {}).get(o.value, 1.0)
        c_w = w.get("complexity", {}).get(l.value, 1.0)
        return round((p or 1) * o_w * c_w, 4)

    @staticmethod
    def _load_rules(path: Path) -> dict[str, Any]:
        if not path.exists(): raise FileNotFoundError(f"Rubric missing at {path}")
        with path.open("r", encoding="utf-8") as fh: return yaml.safe_load(fh)
