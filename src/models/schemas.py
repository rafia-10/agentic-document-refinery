"""
Document Intelligence Refinery — Core Pydantic Schemas
=======================================================
All top-level data models used across the refinery pipeline.

Design principles
-----------------
* Every field carries a concise docstring via `Field(description=...)`.
* Enum guards are used wherever a field is limited to a known vocabulary.
* Bounding boxes are represented as a shared `BoundingBox` value object so
  the shape is consistent everywhere it appears.
* `model_config` is set to `frozen=True` on value-object types and to
  `populate_by_name=True` on all models so both the alias and the Python
  name work in dict-based construction.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared enumerations
# ---------------------------------------------------------------------------


class OriginType(str, Enum):
    """Where the document came from / how it was produced."""

    NATIVE_DIGITAL = "native_digital"     # Born-digital, selectable text
    SCANNED_IMAGE = "scanned_image"       # Image-based scan, needs OCR
    MIXED = "mixed"                      # Combination of digital and scanned
    HTML = "html"                        # Web page or HTML export
    DOCX = "docx"                        # Microsoft Word document
    MARKDOWN = "markdown"                # Plain Markdown
    SPREADSHEET = "spreadsheet"          # CSV, XLSX, or similar
    EMAIL = "email"                      # RFC-822 / MIME email
    UNKNOWN = "unknown"                  # Could not be determined


class LayoutComplexity(str, Enum):
    """Rough measure of how difficult the layout is to parse."""

    SIMPLE = "simple"          # Single-column, no tables, no figures
    MODERATE = "moderate"      # Multi-column or occasional tables
    COMPLEX = "complex"        # Dense tables, figures, footnotes, sidebars
    HIGHLY_COMPLEX = "highly_complex"  # Mixed media, forms, nested structures


class LDUType(str, Enum):
    """Semantic role of a Logical Document Unit."""

    TITLE = "title"
    ABSTRACT = "abstract"
    SECTION_HEADING = "section_heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    CODE_BLOCK = "code_block"
    EQUATION = "equation"
    OTHER = "other"


class DataTypePresent(str, Enum):
    """High-level data modalities present in a section or page."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    EQUATION = "equation"
    CODE = "code"
    IMAGE = "image"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box in normalised page coordinates [0, 1].

    Origin is the top-left corner of the page.  All values must be in
    [0.0, 1.0] with x0 < x1 and y0 < y1.
    """

    model_config = {"frozen": True, "populate_by_name": True}

    x0: float = Field(..., ge=0.0, le=1.0, description="Left edge (normalised)")
    y0: float = Field(..., ge=0.0, le=1.0, description="Top edge (normalised)")
    x1: float = Field(..., ge=0.0, le=1.0, description="Right edge (normalised)")
    y1: float = Field(..., ge=0.0, le=1.0, description="Bottom edge (normalised)")

    @model_validator(mode="after")
    def _validate_order(self) -> "BoundingBox":
        if self.x0 >= self.x1:
            raise ValueError("x0 must be strictly less than x1")
        if self.y0 >= self.y1:
            raise ValueError("y0 must be strictly less than y1")
        return self

    @property
    def area(self) -> float:
        """Normalised area of the bounding box."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)


# ---------------------------------------------------------------------------
# DocumentProfile
# ---------------------------------------------------------------------------


class DocumentProfile(BaseModel):
    """
    Lightweight fingerprint of a document captured *before* deep extraction.

    The profile is derived from cheap heuristics and metadata inspection so
    it can be produced quickly.  Downstream components use it to route the
    document to appropriate extraction strategies and to budget compute.
    """

    model_config = {"populate_by_name": True}

    # --- Identity ---
    document_id: str = Field(
        ...,
        description="Unique identifier for this document (e.g., SHA-256 of raw bytes).",
    )
    filename: str = Field(
        ...,
        description="Original filename including extension.",
    )

    # --- Origin & structure ---
    origin_type: OriginType = Field(
        ...,
        description="How the document was produced; drives OCR and parser selection.",
    )
    layout_complexity: LayoutComplexity = Field(
        ...,
        description=(
            "Estimated structural complexity of the document layout.  "
            "Used to choose chunking strategy and warn about extraction confidence."
        ),
    )
    page_count: Optional[int] = Field(
        None,
        ge=1,
        description="Total number of pages (None if not applicable, e.g., HTML).",
    )

    # --- Linguistic profile ---
    language: str = Field(
        ...,
        description=(
            "BCP-47 language tag of the primary language detected (e.g., 'en', 'fr', 'zh-Hans')."
        ),
    )
    secondary_languages: list[str] = Field(
        default_factory=list,
        description="Other BCP-47 language tags present in smaller portions of the document.",
    )

    # --- Domain hints ---
    domain_hints: list[str] = Field(
        default_factory=list,
        description=(
            "Free-form domain tags inferred from content or metadata, e.g. "
            "['legal', 'finance', 'medical'].  Not exhaustive; used to bias "
            "entity recognisers and summarisers."
        ),
    )

    # --- Cost estimate ---
    estimated_extraction_cost: float = Field(
        ...,
        ge=0.0,
        description=(
            "Rough computational cost in arbitrary units (e.g. normalised GPU-seconds).  "
            "Derived from page count × layout complexity × origin type heuristics."
        ),
    )
    ocr_required: bool = Field(
        False,
        description="True when the document contains image-based text that needs OCR.",
    )

    # --- Optional metadata pass-through ---
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw key-value metadata extracted from the file (PDF XMP, DOCX core props, etc.).",
    )


# ---------------------------------------------------------------------------
# ExtractedDocument
# ---------------------------------------------------------------------------


class TableData(BaseModel):
    """Structured representation of a single extracted table."""

    model_config = {"populate_by_name": True}

    table_id: str = Field(..., description="Unique ID within the parent document.")
    page_references: list[int] = Field(
        ...,
        description="1-indexed page numbers on which this table appears.",
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Location of the table on the page (first page if multi-page).",
    )
    headers: list[str] = Field(
        default_factory=list,
        description="Column headers in left-to-right order.",
    )
    rows: list[list[Any]] = Field(
        default_factory=list,
        description="Row data; each inner list corresponds to one row.",
    )
    caption: Optional[str] = Field(None, description="Table caption if present.")
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score [0, 1].",
    )


class FigureData(BaseModel):
    """Metadata and content associated with an extracted figure."""

    model_config = {"populate_by_name": True}

    figure_id: str = Field(..., description="Unique ID within the parent document.")
    page_references: list[int] = Field(
        ...,
        description="1-indexed page numbers containing this figure.",
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Location of the figure on the page.",
    )
    caption: Optional[str] = Field(None, description="Figure caption if present.")
    alt_text: Optional[str] = Field(
        None,
        description="Alt text or OCR-derived description of the figure contents.",
    )
    image_path: Optional[str] = Field(
        None,
        description="Path to the extracted image file (if saved to disk).",
    )


class ExtractedDocument(BaseModel):
    """
    Normalised, flat representation of everything extracted from a document.

    This is the primary output of the extraction stage.  It contains raw
    content in a structured form before any semantic chunking or indexing.
    """

    model_config = {"populate_by_name": True}

    # --- Identity ---
    document_id: str = Field(
        ...,
        description="Must match the `document_id` in the corresponding DocumentProfile.",
    )
    extraction_version: str = Field(
        "1.0.0",
        description="SemVer string of the extraction pipeline that produced this record.",
    )

    # --- Full text ---
    full_text: str = Field(
        ...,
        description=(
            "Complete extracted text of the document in reading order.  "
            "Includes body text but excludes headers/footers unless indistinguishable."
        ),
    )
    full_text_by_page: list[str] = Field(
        default_factory=list,
        description=(
            "Per-page extracted text.  Index 0 = page 1.  "
            "Empty string for pages with no extractable text."
        ),
    )

    # --- Structured elements ---
    tables: list[TableData] = Field(
        default_factory=list,
        description="All tables found in the document.",
    )
    figures: list[FigureData] = Field(
        default_factory=list,
        description="All figures/images found in the document.",
    )

    # --- Spatial index ---
    bounding_boxes: dict[str, BoundingBox] = Field(
        default_factory=dict,
        description=(
            "Mapping from element ID (e.g., table_id, figure_id, or a span ID) "
            "to its bounding box on the page.  Enables spatial queries."
        ),
    )

    # --- Page references ---
    page_count: int = Field(
        ...,
        ge=1,
        description="Number of pages processed.",
    )
    page_dimensions: dict[int, tuple[float, float]] = Field(
        default_factory=dict,
        description=(
            "Physical page dimensions in points (width, height) keyed by 1-indexed page number."
        ),
    )

    # --- Quality signals ---
    overall_confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Aggregate extraction confidence across all modalities.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during extraction.",
    )


# ---------------------------------------------------------------------------
# LDU — Logical Document Unit
# ---------------------------------------------------------------------------


class LDU(BaseModel):
    """
    A single, semantically coherent chunk of a document.

    LDUs are the primary unit consumed by downstream tasks (embedding,
    retrieval, QA).  They are produced by the chunking stage from an
    ExtractedDocument.
    """

    model_config = {"populate_by_name": True}

    # --- Identity ---
    ldu_id: str = Field(
        ...,
        description="Unique ID for this unit within the document.",
    )
    document_id: str = Field(
        ...,
        description="ID of the parent document (links back to ExtractedDocument).",
    )

    # --- Content ---
    content: str = Field(
        ...,
        description="The actual text content of this logical unit.",
    )
    ldu_type: LDUType = Field(
        ...,
        description="Semantic role of this unit (heading, paragraph, table, …).",
    )

    # --- Token budget ---
    token_count: int = Field(
        ...,
        ge=0,
        description=(
            "Approximate token count for the content field under the target tokeniser.  "
            "Used for context-window budget management."
        ),
    )

    # --- Content hash ---
    content_hash: str = Field(
        ...,
        description=(
            "SHA-256 hex digest of the UTF-8 encoded `content` string.  "
            "Enables deduplication and change detection."
        ),
    )

    # --- Provenance / spatial ---
    page_references: list[int] = Field(
        ...,
        description="1-indexed page numbers this LDU spans.",
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Location of this LDU on the page (first page if multi-page).",
    )

    # --- Document structure ---
    parent_section: Optional[str] = Field(
        None,
        description=(
            "Title or ID of the nearest ancestor section heading.  "
            "None for top-level content or documents without headings."
        ),
    )
    sequence_index: int = Field(
        ...,
        ge=0,
        description="0-based position of this LDU in the document reading order.",
    )

    # --- Optional enrichment ---
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs added by enrichment passes (e.g., NER results).",
    )

    @model_validator(mode="before")
    @classmethod
    def _auto_content_hash(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Auto-compute content_hash if not supplied."""
        content = values.get("content")
        if content and not values.get("content_hash"):
            values["content_hash"] = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()
        return values


# ---------------------------------------------------------------------------
# PageIndex
# ---------------------------------------------------------------------------


class SectionNode(BaseModel):
    """
    One node in the hierarchical section tree of a document.

    Nodes can be nested arbitrarily via `children` to represent a full
    chapter → section → sub-section hierarchy.
    """

    model_config = {"populate_by_name": True}

    section_id: str = Field(..., description="Unique ID for this section node.")
    title: str = Field(..., description="Section heading text.")
    level: int = Field(
        ...,
        ge=1,
        description="Heading depth (1 = top-level chapter, 2 = section, etc.).",
    )
    page_references: list[int] = Field(
        ...,
        description="1-indexed page numbers covered by this section.",
    )
    summary: Optional[str] = Field(
        None,
        description="Auto-generated or human-written summary of this section's content.",
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description=(
            "Named entities, concepts, or keywords considered most salient in this section."
        ),
    )
    data_types_present: list[DataTypePresent] = Field(
        default_factory=list,
        description="Content modalities detected within this section.",
    )
    ldu_ids: list[str] = Field(
        default_factory=list,
        description="Ordered IDs of LDUs that belong directly to this section.",
    )
    children: list["SectionNode"] = Field(
        default_factory=list,
        description="Nested sub-sections (recursive).",
    )


# Forward reference resolution for recursive model
SectionNode.model_rebuild()


class PageIndex(BaseModel):
    """
    Full structural index of a document expressed as a section tree.

    The PageIndex is built after chunking and is the primary artefact used
    for hierarchical navigation, targeted retrieval, and audit trails.
    """

    model_config = {"populate_by_name": True}

    document_id: str = Field(
        ...,
        description="ID of the document this index belongs to.",
    )
    index_version: str = Field(
        "1.0.0",
        description="SemVer of the indexing pipeline that produced this record.",
    )

    # --- Top-level tree ---
    sections: list[SectionNode] = Field(
        ...,
        description="Top-level section nodes; each may contain nested children.",
    )

    # --- Document-level aggregates ---
    document_summary: Optional[str] = Field(
        None,
        description="High-level summary of the entire document.",
    )
    global_key_entities: list[str] = Field(
        default_factory=list,
        description="Most salient entities across the whole document.",
    )
    global_data_types: list[DataTypePresent] = Field(
        default_factory=list,
        description="Union of all data modalities present anywhere in the document.",
    )

    # --- Flat look-up helpers ---
    ldu_to_section: dict[str, str] = Field(
        default_factory=dict,
        description="Map from ldu_id → section_id for O(1) parent lookup.",
    )
    page_to_sections: dict[int, list[str]] = Field(
        default_factory=dict,
        description="Map from 1-indexed page number → list of section_ids present on that page.",
    )


# ---------------------------------------------------------------------------
# ProvenanceChain
# ---------------------------------------------------------------------------


class ProvenanceRecord(BaseModel):
    """
    A single link in a provenance chain: one source span that contributed
    to a derived piece of information.
    """

    model_config = {"frozen": True, "populate_by_name": True}

    document_name: str = Field(
        ...,
        description="Human-readable filename or title of the source document.",
    )
    document_id: str = Field(
        ...,
        description="Machine ID of the source document (matches DocumentProfile.document_id).",
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page containing the source span.",
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Exact location of the source span on the page.",
    )
    content_hash: str = Field(
        ...,
        description=(
            "SHA-256 hex digest of the source text span.  "
            "Used to verify that the referenced content has not changed."
        ),
    )
    excerpt: Optional[str] = Field(
        None,
        description="Short verbatim excerpt from the source for human inspection.",
    )


class ProvenanceChain(BaseModel):
    """
    Ordered sequence of provenance records for a derived artefact.

    Tracks *every* source document span that contributed to a claim,
    answer, summary, or other derived output so that answers can always
    be traced back to their origins.
    """

    model_config = {"populate_by_name": True}

    chain_id: str = Field(
        ...,
        description="Unique ID for this provenance chain.",
    )
    derived_artefact_id: str = Field(
        ...,
        description=(
            "ID of the artefact whose provenance this chain describes "
            "(e.g., an LDU ID, an answer span ID, or a summary ID)."
        ),
    )
    records: list[ProvenanceRecord] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of source records, from most to least relevant.  "
            "Must contain at least one record."
        ),
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text annotation about how the records were combined or weighted.",
    )
