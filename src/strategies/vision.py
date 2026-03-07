"""
VisionExtractor — OpenRouter GPT-4o-mini Vision API
=====================================================
Converts each PDF page (or image file) to a base64-encoded PNG and sends
them to the OpenRouter-hosted GPT-4o-mini vision endpoint using the
standard OpenAI chat-completions API.

Configuration (environment variables)
--------------------------------------
OPENROUTER_API_KEY   Required.  Your OpenRouter API key.
OPENROUTER_BASE_URL  Optional.  Default: https://openrouter.ai/api/v1
OPENROUTER_MODEL     Optional.  Default: openai/gpt-4o-mini

Stub mode (CI / local dev without key)
---------------------------------------
Set ``REFINERY_VISION_STUB=1`` to return a zero-confidence placeholder
without making any API calls.

Install dependencies
--------------------
    pip install "document-refinery[vision]"
    # pulls in: openai>=1.30, pdf2image>=1.17, Pillow>=10.3
    # requires: poppler-utils (for pdf2image)
    #   apt-get install poppler-utils
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from pathlib import Path

from src.utils.llm import get_chat_model
from src.models.schemas import BoundingBox, DocumentProfile, ExtractedDocument
from src.strategies.base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openai/gpt-4o-mini"

_EXTRACTION_PROMPT = (
    "You are a precise document extraction assistant. "
    "Extract ALL visible text from this document page in reading order. "
    "Preserve table structure using Markdown pipe syntax. "
    "Label figures as 'Figure N: <caption>'. "
    "Do NOT summarise, interpret, or add commentary — output the raw content only."
)


class VisionExtractor(BaseExtractor):
    """
    Strategy 3: OpenRouter GPT-4o-mini vision extraction.

    Each page is sent as a base64 PNG to the GPT-4o-mini vision endpoint
    via the OpenAI-compatible OpenRouter API.  Works on scanned PDFs,
    image PDFs, and standalone image files.

    Confidence heuristic
    --------------------
    ``confidence = useful_char_ratio`` of the concatenated extracted text,
    clamped to [0, 1].  A low ratio means the model returned mostly noise
    or refused to extract (e.g., blank/degenerate pages).
    """

    @property
    def name(self) -> str:
        return "vision"

    def extract(
        self,
        file_path: str | Path,
        profile: DocumentProfile,
    ) -> ExtractionResult:
        """
        Run vision extraction via OpenRouter GPT-4o-mini.

        Falls back to stub result when ``REFINERY_VISION_STUB=1``.
        Raises ``RuntimeError`` when ``OPENROUTER_API_KEY`` is not set
        (and stub mode is off) so the misconfiguration is surfaced early.
        """
        if os.getenv("REFINERY_VISION_STUB", "").strip() == "1":
            logger.warning(
                "VisionExtractor: REFINERY_VISION_STUB=1 — returning placeholder."
            )
            return self._stub_result(profile)

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise NotImplementedError(
                "VisionExtractor requires OPENROUTER_API_KEY to be set.\n"
                "Export it with: export OPENROUTER_API_KEY=sk-or-...\n"
                "Or set REFINERY_VISION_STUB=1 to suppress this error."
            )

        self._require_vision_deps()

        path = Path(file_path)
        page_images = self._rasterise(path, profile)

        client = get_chat_model(purpose="vision")
        page_texts: list[str] = []
        warnings: list[str] = []

        for page_no, img_b64 in enumerate(page_images, start=1):
            try:
                # Use LangChain's vision support via get_chat_model
                from langchain_core.messages import HumanMessage
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": _EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ]
                )
                response = client.invoke([message])
                text = response.content
                page_texts.append(text)
            except Exception as exc:
                msg = f"Page {page_no}: vision API call failed — {exc}"
                logger.warning(msg)
                warnings.append(msg)
                page_texts.append("")

        full_text = "\n\n".join(t for t in page_texts if t)
        
        # Spatial Provenance synthesis
        bounding_boxes = self._synthesize_bboxes(page_texts)
        
        confidence = self._compute_enhanced_confidence(page_texts)

        doc = ExtractedDocument(
            document_id=profile.document_id,
            full_text=full_text,
            full_text_by_page=page_texts,
            page_count=len(page_images) or (profile.page_count or 1),
            overall_confidence=confidence,
            warnings=warnings,
            bounding_boxes=bounding_boxes,
        )
        return ExtractionResult(
            document=doc,
            confidence=confidence,
            strategy_name=self.name,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Rasterisation (PDF → PNG pages)
    # ------------------------------------------------------------------

    @staticmethod
    def _rasterise(path: Path, profile: DocumentProfile) -> list[str]:
        """
        Convert document to a list of base64-encoded PNG strings (one per page).

        For PDFs, uses pdf2image (requires poppler-utils on the host).
        For image files (PNG, JPEG), wraps the single file directly.
        """
        suffix = path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            raw = path.read_bytes()
            return [base64.b64encode(raw).decode("ascii")]

        # PDF → image conversion
        from pdf2image import convert_from_path  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]  # noqa: F401

        images = convert_from_path(str(path), dpi=200, fmt="png")
        result: list[str] = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        return result

    # ------------------------------------------------------------------
    # OpenRouter API call
    # ------------------------------------------------------------------

    @staticmethod
    def _make_client(api_key: str):  # type: ignore[no-untyped-def]
        from openai import OpenAI  # type: ignore[import-not-found]

        base_url = os.getenv("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        return OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _call_vision_api(client, img_b64: str) -> str:  # type: ignore[no-untyped-def]
        """Send one page image to GPT-4o-mini and return extracted text."""
        model = os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _EXTRACTION_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Spatial Synthesis
    # ------------------------------------------------------------------

    @staticmethod
    def _synthesize_bboxes(page_texts: list[str]) -> dict[str, BoundingBox]:
        """Produce approximate line-level bboxes for spatial consistency."""
        boxes: dict[str, BoundingBox] = {}
        for p_idx, text in enumerate(page_texts):
            lines = [l for l in text.splitlines() if l.strip()]
            if not lines: continue
            lh = 1.0 / len(lines)
            for l_idx, _ in enumerate(lines):
                boxes[f"v-p{p_idx:03d}-l{l_idx:03d}"] = BoundingBox(x0=0, y0=l_idx*lh, x1=1, y1=(l_idx+1)*lh)
        return boxes

    # ------------------------------------------------------------------
    # Enhanced Confidence & Refusal Detection
    # ------------------------------------------------------------------

    @classmethod
    def _compute_enhanced_confidence(cls, page_texts: list[str]) -> float:
        """
        Aggregate confidence across pages, penalising refusals.
        """
        if not page_texts: return 0.0
        
        page_scores: list[float] = []
        for text in page_texts:
            if cls._is_refusal(text):
                page_scores.append(0.0)
            else:
                page_scores.append(cls._compute_page_confidence(text))
        
        return round(sum(page_scores) / len(page_scores), 4)

    @staticmethod
    def _is_refusal(text: str) -> bool:
        """Detect common LLM refusal patterns for vision tasks."""
        refusal_patterns = [
            r"i (cannot|can't) (read|see|extract)",
            r"blank page",
            r"not (possible|able) to",
            r"sorry, but as an ai",
        ]
        low_text = text.lower()
        return any(re.search(p, low_text) for p in refusal_patterns)

    @staticmethod
    def _compute_page_confidence(text: str) -> float:
        """Useful-char ratio: alphanumerics + punctuation / total chars."""
        if not text: return 0.0
        useful = len(re.findall(r"[A-Za-z0-9.,;:!?'\"\-\(\)\[\]]", text))
        return min(useful / len(text), 1.0)

    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------

    @staticmethod
    def _require_vision_deps() -> None:
        missing: list[str] = []
        for pkg in ("openai", "pdf2image", "PIL"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise RuntimeError(
                f"VisionExtractor is missing required packages: {missing}.\n"
                "Install with: pip install \"document-refinery[vision]\""
            )

    # ------------------------------------------------------------------
    # Stub / fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_result(profile: DocumentProfile) -> ExtractionResult:
        doc = ExtractedDocument(
            document_id=profile.document_id,
            full_text="",
            page_count=profile.page_count or 1,
            overall_confidence=0.0,
            warnings=[
                "VisionExtractor stub mode active (REFINERY_VISION_STUB=1). "
                "Set OPENROUTER_API_KEY and remove the stub flag for real extraction."
            ],
        )
        return ExtractionResult(
            document=doc,
            confidence=0.0,
            strategy_name="vision",
            warnings=doc.warnings,
        )
