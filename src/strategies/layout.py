"""
LayoutExtractor — Docling-Powered Layout-Aware Extraction
==========================================================
Uses IBM's `docling` library to produce a rich structured representation
of any supported document format (PDF, DOCX, PPTX, HTML, images).

Docling gives us:
  * Per-page text with reading-order preservation
  * Detected tables serialised to Markdown or JSON
  * Figure / image metadata with captions
  * Section headings with hierarchy levels
  * Normalised bounding boxes for every text block

Confidence heuristic
--------------------
``confidence = covered_bbox_area / (page_count × 0.90)``

If docling is not installed, the extractor degrades gracefully to the
FastTextExtractor heuristic path and logs a warning.

Install docling
---------------
    pip install docling
    # or: pip install "document-refinery[layout]"
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.models.schemas import (
    BoundingBox,
    DocumentProfile,
    ExtractedDocument,
    FigureData,
    OriginType,
    TableData,
)
from src.strategies.base import BaseExtractor, ExtractionResult
from src.strategies.fast_text import FastTextExtractor

logger = logging.getLogger(__name__)


def _docling_available() -> bool:
    try:
        import docling  # noqa: F401  # type: ignore[import-not-found]
        return True
    except ImportError:
        return False


class LayoutExtractor(BaseExtractor):
    """
    Strategy 2: Docling-powered layout-aware extraction.

    Improvements over FastTextExtractor
    ------------------------------------
    * Structure-aware reading order (no column bleed)
    * Native table extraction — returns Markdown or structured rows
    * Figure metadata with captions
    * Accurate bounding boxes from the docling layout engine
    * Handles PDF, DOCX, PPTX, HTML, PNG/JPEG images

    Fallback
    --------
    When `docling` is not installed, delegates to :class:`FastTextExtractor`
    with a capped confidence of 0.65 (below the layout threshold) to allow
    escalation decisions to remain correct.
    """

    @property
    def name(self) -> str:
        return "layout"

    def extract(
        self,
        file_path: str | Path,
        profile: DocumentProfile,
    ) -> ExtractionResult:
        path = Path(file_path)

        if _docling_available():
            return self._extract_with_docling(path, profile)

        logger.warning(
            "docling is not installed; falling back to FastTextExtractor. "
            "Install with: pip install docling"
        )
        return self._fallback(path, profile)

    # ------------------------------------------------------------------
    # Docling path
    # ------------------------------------------------------------------

    def _extract_with_docling(
        self,
        path: Path,
        profile: DocumentProfile,
    ) -> ExtractionResult:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

        warnings: list[str] = []
        converter = DocumentConverter()

        try:
            result = converter.convert(str(path))
        except Exception as exc:
            warnings.append(f"docling conversion error: {exc}; falling back to FastText.")
            return self._fallback(path, profile, extra_warnings=warnings)

        doc_obj = result.document

        # ---- Full text (reading-order preserved) -------------------------
        full_text: str = doc_obj.export_to_markdown()

        # ---- Per-page text -----------------------------------------------
        page_texts: list[str] = self._collect_page_texts(doc_obj)

        # ---- Tables --------------------------------------------------------
        tables: list[TableData] = self._collect_tables(doc_obj)

        # ---- Figures -------------------------------------------------------
        figures: list[FigureData] = self._collect_figures(doc_obj)

        # ---- Bounding boxes ------------------------------------------------
        bounding_boxes: dict[str, BoundingBox] = self._collect_bboxes(doc_obj)

        # ---- Confidence ---------------------------------------------------
        page_count = max(len(page_texts), profile.page_count or 1)
        confidence = self._compute_confidence(bounding_boxes, page_count)

        if profile.origin_type == OriginType.SCANNED_IMAGE:
            warnings.append(
                "LayoutExtractor: scanned PDF detected — docling will run its "
                "internal OCR (if enabled); consider VisionExtractor for higher accuracy."
            )
            confidence = min(confidence, 0.70)

        doc = ExtractedDocument(
            document_id=profile.document_id,
            full_text=full_text,
            full_text_by_page=page_texts,
            tables=tables,
            figures=figures,
            bounding_boxes=bounding_boxes,
            page_count=page_count,
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
    # Docling object → Refinery models
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_page_texts(doc_obj) -> list[str]:  # type: ignore[no-untyped-def]
        """
        Group docling text elements by page number.

        Docling's DoclingDocument exposes `.pages` and `.texts` iterables.
        We fall back to a single-page grouping if the attribute is absent.
        """
        try:
            page_map: dict[int, list[str]] = {}
            for item, _ in doc_obj.iterate_items():
                prov = getattr(item, "prov", None)
                if prov:
                    page_no = prov[0].page_no if prov else 1
                else:
                    page_no = 1
                text = getattr(item, "text", None) or ""
                if text.strip():
                    page_map.setdefault(page_no, []).append(text)
            if not page_map:
                return [doc_obj.export_to_markdown()]
            return ["\n".join(page_map.get(p, [])) for p in sorted(page_map)]
        except Exception:
            return [doc_obj.export_to_markdown()]

    @staticmethod
    def _collect_tables(doc_obj) -> list[TableData]:  # type: ignore[no-untyped-def]
        """Extract tables from docling's TableItem list."""
        tables: list[TableData] = []
        table_idx = 0
        try:
            for table_item, _ in doc_obj.iterate_items():
                if type(table_item).__name__ != "TableItem":
                    continue
                table_idx += 1
                try:
                    df = table_item.export_to_dataframe()
                    headers = list(df.columns.astype(str))
                    rows = [list(row.astype(str)) for _, row in df.iterrows()]
                except Exception:
                    md = table_item.export_to_markdown()
                    lines = [l for l in md.splitlines() if "|" in l and "---" not in l]
                    if not lines:
                        continue
                    headers = [c.strip() for c in lines[0].split("|") if c.strip()]
                    rows = [
                        [c.strip() for c in line.split("|") if c.strip()]
                        for line in lines[1:]
                    ]

                prov = getattr(table_item, "prov", None)
                page_refs = [prov[0].page_no] if prov else [1]

                tables.append(
                    TableData(
                        table_id=f"tbl-{table_idx:04d}",
                        page_references=page_refs,
                        headers=headers,
                        rows=rows,
                        confidence=0.90,
                    )
                )
        except Exception as exc:
            logger.debug("Table extraction from docling failed: %s", exc)
        return tables

    @staticmethod
    def _collect_figures(doc_obj) -> list[FigureData]:  # type: ignore[no-untyped-def]
        """Extract figure/image metadata from docling."""
        figures: list[FigureData] = []
        fig_idx = 0
        try:
            for item, _ in doc_obj.iterate_items():
                if type(item).__name__ not in ("PictureItem", "FigureItem"):
                    continue
                fig_idx += 1
                caption = ""
                if hasattr(item, "captions") and item.captions:
                    caption = " ".join(
                        getattr(c, "text", str(c)) for c in item.captions
                    )
                prov = getattr(item, "prov", None)
                page_refs = [prov[0].page_no] if prov else [1]
                figures.append(
                    FigureData(
                        figure_id=f"fig-{fig_idx:04d}",
                        page_references=page_refs,
                        caption=caption,
                    )
                )
        except Exception as exc:
            logger.debug("Figure extraction from docling failed: %s", exc)
        return figures

    @staticmethod
    def _collect_bboxes(doc_obj) -> dict[str, BoundingBox]:  # type: ignore[no-untyped-def]
        """
        Convert docling provenance bounding boxes to normalised BoundingBox objects.

        Docling bbox coords are fractional [0,1] relative to page dimensions
        once accessed via ``prov.bbox`` (BoundingBox in docling's own model).
        """
        boxes: dict[str, BoundingBox] = {}
        try:
            for idx, (item, _) in enumerate(doc_obj.iterate_items()):
                prov = getattr(item, "prov", None)
                if not prov:
                    continue
                for p_idx, p in enumerate(prov):
                    bbox = getattr(p, "bbox", None)
                    if bbox is None:
                        continue
                    # docling bbox: l, t, r, b (all normalised 0-1)
                    try:
                        span_id = f"item-{idx:05d}-prov-{p_idx}"
                        boxes[span_id] = BoundingBox(
                            x0=float(bbox.l),
                            y0=float(bbox.t),
                            x1=float(bbox.r),
                            y1=float(bbox.b),
                        )
                    except (AttributeError, ValueError):
                        pass
        except Exception as exc:
            logger.debug("Bounding box collection failed: %s", exc)
        return boxes

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        bounding_boxes: dict[str, BoundingBox],
        page_count: int,
    ) -> float:
        if page_count == 0:
            return 0.0
        total_area = sum(bb.area for bb in bounding_boxes.values())
        return round(min(total_area / (0.90 * page_count), 1.0), 4)

    # ------------------------------------------------------------------
    # Fallback path (docling unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(
        path: Path,
        profile: DocumentProfile,
        extra_warnings: list[str] | None = None,
    ) -> ExtractionResult:
        fast = FastTextExtractor()
        result = fast.extract(path, profile)
        capped_conf = min(result.confidence, 0.65)
        warns = list(extra_warnings or []) + result.warnings + [
            "LayoutExtractor: docling not available; running FastTextExtractor fallback. "
            "Install docling: pip install docling"
        ]
        return ExtractionResult(
            document=result.document,
            confidence=capped_conf,
            strategy_name="layout(fallback:fast_text)",
            warnings=warns,
        )
