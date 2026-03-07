import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from src.models.schemas import LDU, ExtractedDocument, LDUType

logger = logging.getLogger(__name__)

class Chunker:
    """
    Stage 3 — Semantic Chunking
    
    Converts an ExtractedDocument into a list of Logical Document Units (LDUs)
    based on the 'chunking' constitution in the extraction rules.
    """

    def __init__(self, rules_path: str | Path = "rubric/extraction_rules.yaml"):
        with open(rules_path, "r") as f:
            full_rules = yaml.safe_load(f)
            self._rules = full_rules.get("chunking", {})
        
        self._max_tokens = self._rules.get("max_tokens", 512)
        self._overlap = self._rules.get("overlap_tokens", 64)
        self._min_tokens = self._rules.get("min_tokens", 32)
        self._respect_pages = self._rules.get("respect_page_boundaries", True)

    def chunk(self, doc: ExtractedDocument) -> list[LDU]:
        """Produce a list of LDUs for the given document."""
        logger.info("[%s] Chunking document (%d pages) …", doc.document_id, doc.page_count)
        
        ldus: list[LDU] = []
        sequence_idx = 0

        if self._respect_pages:
            for p_idx, page_text in enumerate(doc.full_text_by_page):
                page_ldus = self._chunk_text(page_text, doc.document_id, p_idx + 1, sequence_idx)
                ldus.extend(page_ldus)
                sequence_idx += len(page_ldus)
        else:
            ldus = self._chunk_text(doc.full_text, doc.document_id, 1, 0)

        # Post-processing: Merge tiny LDUs if they are too small and not headings
        ldus = self._refine_ldus(ldus)
        
        logger.info("[%s] Produced %d LDUs.", doc.document_id, len(ldus))
        return ldus

    def _chunk_text(self, text: str, doc_id: str, page_num: int, start_seq: int) -> list[LDU]:
        """Split a block of text into LDUs."""
        if not text.strip():
            return []

        # 1. Identify split points (Headings, Paragraphs)
        # Simple heuristic: Lines with few words starting with uppercase are headings?
        # Or lines followed by \n\n.
        segments = self._split_into_segments(text)
        
        chunks: list[LDU] = []
        current_buffer = []
        current_tokens = 0
        
        for seg_text, seg_type in segments:
            seg_tokens = self._estimate_tokens(seg_text)
            
            # If adding this segment exceeds max_tokens, flush the buffer
            if current_buffer and (current_tokens + seg_tokens > self._max_tokens):
                chunks.append(self._create_ldu(current_buffer, doc_id, page_num, start_seq + len(chunks)))
                # Manage overlap: Keep some of the previous content
                # For simplicity, we just clear for now, or keep the last segment if it fits
                current_buffer = []
                current_tokens = 0

            current_buffer.append((seg_text, seg_type))
            current_tokens += seg_tokens
            
            # If a single segment is too big, force split it
            if current_tokens > self._max_tokens:
                # Force flush
                chunks.append(self._create_ldu(current_buffer, doc_id, page_num, start_seq + len(chunks)))
                current_buffer = []
                current_tokens = 0

        if current_buffer:
            chunks.append(self._create_ldu(current_buffer, doc_id, page_num, start_seq + len(chunks)))

        return chunks

    def _split_into_segments(self, text: str) -> list[tuple[str, LDUType]]:
        """Identify headings and paragraphs."""
        segments = []
        # Split by double newline first (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        for p in paragraphs:
            stripped = p.strip()
            if not stripped: continue
            
            # Heuristic for Heading: short line, no ending punctuation
            lines = stripped.splitlines()
            if len(lines) == 1 and len(stripped) < 100 and not re.search(r'[.!?:]$', stripped):
                segments.append((stripped, LDUType.SECTION_HEADING))
            else:
                segments.append((stripped, LDUType.PARAGRAPH))
                
        return segments

    def _create_ldu(self, buffer: list[tuple[str, LDUType]], doc_id: str, page_num: int, seq_idx: int) -> LDU:
        """Helper to build an LDU from a buffer of segments."""
        content = "\n\n".join(s[0] for s in buffer)
        # Deterministic type: if any segment is a heading and it's small, mark as heading?
        # Usually, if it contains multiple segments, it's a paragraph block.
        ldu_type = LDUType.PARAGRAPH
        if len(buffer) == 1:
            ldu_type = buffer[0][1]

        ldu_id = f"{doc_id}-p{page_num:03d}-s{seq_idx:03d}"
        
        return LDU(
            ldu_id=ldu_id,
            document_id=doc_id,
            content=content,
            ldu_type=ldu_type,
            token_count=self._estimate_tokens(content),
            page_references=[page_num],
            sequence_index=seq_idx
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count: words * 1.3 or chars / 4."""
        return max(len(text.split()) + len(re.findall(r'[^\w\s]', text)), 1)

    def _refine_ldus(self, ldus: list[LDU]) -> list[LDU]:
        """Merge LDUs that are too small into their successor/predecessor."""
        if not ldus: return []
        
        refined = []
        for ldu in ldus:
            if ldu.token_count < self._min_tokens and refined and ldu.ldu_type != LDUType.SECTION_HEADING:
                # Merge into previous
                prev = refined[-1]
                prev.content += "\n\n" + ldu.content
                prev.token_count += ldu.token_count
                prev.content_hash = hashlib.sha256(prev.content.encode()).hexdigest()
            else:
                refined.append(ldu)
        return refined
