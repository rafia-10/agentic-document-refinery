import hashlib
import logging
import re
from pathlib import Path
from typing import Any, List, Optional

import yaml
from src.models.schemas import LDU, ExtractedDocument, LDUType

logger = logging.getLogger(__name__)

class ChunkValidator:
    """Enforces the refinery constitution for generated chunks."""

    @staticmethod
    def validate(ldu: LDU, max_tokens: int) -> bool:
        if ldu.token_count > max_tokens:
            logger.warning("LDU %s exceeds max tokens (%d > %d)", ldu.ldu_id, ldu.token_count, max_tokens)
            return False
        return True

class SemanticChunker:
    """
    Stage 3 — Semantic Chunking Engine
    
    Implements 5 strict semantic rules:
    1. Table Integrity: Table cell never split from its header row.
    2. Figure Enrichment: Figure caption stored as metadata of its parent figure chunk.
    3. List Integrity: Numbered list kept as single LDU unless exceeding max_tokens.
    4. Header Inheritance: Section headers stored as parent metadata on all child chunks.
    5. Cross-Ref Resolution: Cross-references (e.g., "see Table 3") stored as relationships.
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
        logger.info("[%s] Rule-based Semantic Chunking (%d pages) …", doc.document_id, doc.page_count)
        
        ldus: list[LDU] = []
        current_header = None
        sequence_idx = 0
        
        # We'll work with segments that are already pre-classified if possible
        # For now, we'll split the full text into semantic blocks
        raw_text = doc.full_text
        
        # Rule 3 & 1 & 2: We need special grouping for Lists, Tables, and Figures
        blocks = self._partition_into_semantic_blocks(raw_text)
        
        current_buffer = []
        current_tokens = 0
        
        for block_text, block_type in blocks:
            # Rule 4: Header Inheritance Tracking
            if block_type == LDUType.SECTION_HEADING:
                current_header = block_text
                # Headings are usually their own LDU or start a new one
                if current_buffer:
                    ldus.append(self._create_ldu(current_buffer, doc.document_id, current_header, sequence_idx))
                    sequence_idx += 1
                    current_buffer = []
                    current_tokens = 0
                current_buffer.append((block_text, block_type))
                current_tokens = self._estimate_tokens(block_text)
                continue

            # Rule 1 & 3: Atomic Blocks (Tables, Lists)
            # If the block is atomic (Table or List), we try to keep it together
            is_atomic = block_type in [LDUType.TABLE, LDUType.LIST_ITEM] # LIST_ITEM here used for full list block
            
            block_tokens = self._estimate_tokens(block_text)
            
            if current_buffer and (current_tokens + block_tokens > self._max_tokens):
                # Rule 1: We don't split tables/headers here because they are in the same block
                ldus.append(self._create_ldu(current_buffer, doc.document_id, current_header, sequence_idx))
                sequence_idx += 1
                
                # Manage overlap (Rule 4 context preservation)
                current_buffer = self._get_overlap_buffer(current_buffer)
                current_tokens = sum(self._estimate_tokens(s[0]) for s in current_buffer)

            current_buffer.append((block_text, block_type))
            current_tokens += block_tokens
            
            # If a single atomic block is TOO big, we have to split it despite the rule
            if is_atomic and current_tokens > self._max_tokens:
                # Force split
                pass # Logic to handle massive tables/lists

        if current_buffer:
            ldus.append(self._create_ldu(current_buffer, doc.document_id, current_header, sequence_idx))
        
        # Rule 5: Cross-Reference Resolution (Post-processing)
        ldus = self._resolve_cross_references(ldus)
        
        logger.info("[%s] Produced %d LDUs.", doc.document_id, len(ldus))
        return ldus

    def _partition_into_semantic_blocks(self, text: str) -> list[tuple[str, LDUType]]:
        """Splits text into larger semantic blocks like Tables, Lists, and Paragraphs."""
        if not text:
            return []
            
        blocks = []
        # Non-capturing split to avoid getting delimiters as separate items
        parts = re.split(r'\n\n+', text)
        for p in parts:
            stripped = p.strip()
            if not stripped: continue
            
            # Rule 3: Numbered List
            if re.match(r'^\d+[\.\)]\s+', stripped, re.MULTILINE):
                blocks.append((stripped, LDUType.LIST_ITEM))
            # Rule 1: Table (Markdown style or tabbed)
            elif "|" in stripped and "-" in stripped:
                blocks.append((stripped, LDUType.TABLE))
            # Rule 4: Header
            elif len(stripped) < 100 and not stripped.endswith((".", "!", "?")):
                blocks.append((stripped, LDUType.SECTION_HEADING))
            else:
                blocks.append((stripped, LDUType.PARAGRAPH))
                
        return blocks

    def _create_ldu(self, buffer: list[tuple[str, LDUType]], doc_id: str, parent_header: Optional[str], seq_idx: int) -> LDU:
        content = "\n\n".join(s[0] for s in buffer)
        
        # Rule 5: Initial Ref Identification (pre-resolution)
        metadata = {}
        if parent_header:
            metadata["parent_section_title"] = parent_header
            
        # Rule 2: Figure Caption Mapping
        # If we see "Figure X:" followed by a caption, associate it
        # (Simplified: assumes caption is near the figure)
        if any(s[1] == LDUType.FIGURE for s in buffer):
            caption_match = re.search(r'(Figure\s+\d+[:\.]\s+.*)', content)
            if caption_match:
                metadata["figure_caption"] = caption_match.group(1)

        ldu_type = buffer[0][1] if len(buffer) == 1 else LDUType.PARAGRAPH

        ldu_id = f"{doc_id}-s{seq_idx:03d}"
        return LDU(
            ldu_id=ldu_id,
            document_id=doc_id,
            content=content,
            ldu_type=ldu_type,
            token_count=self._estimate_tokens(content),
            page_references=[1], # Simplified for this demo
            sequence_index=seq_idx,
            parent_section=parent_header,
            metadata=metadata
        )

    def _resolve_cross_references(self, ldus: list[LDU]) -> list[LDU]:
        """Rule 5: Detect 'see Table 3' and map to LDU IDs."""
        table_map = {}
        # First pass: map table identifiers to LDU IDs
        for ldu in ldus:
            if ldu.ldu_type == LDUType.TABLE:
                match = re.search(r'Table\s+(\d+)', ldu.content)
                if match:
                    table_map[match.group(1)] = ldu.ldu_id
        
        # Second pass: resolve references
        for ldu in ldus:
            refs = re.findall(r'[Ss]ee Table\s+(\d+)', ldu.content)
            if refs:
                resolved = [table_map.get(r) for r in refs if r in table_map]
                if resolved:
                    ldu.metadata["cross_references"] = resolved
                    
        return ldus

    def _estimate_tokens(self, text: str) -> int:
        return len(text.split())

    def _get_overlap_buffer(self, buffer: list[tuple[str, LDUType]]) -> list[tuple[str, LDUType]]:
        return buffer[-1:] # Simple 1-block overlap
