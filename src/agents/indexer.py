import logging
from typing import Any
from src.models.schemas import PageIndex, SectionNode, LDU, ExtractedDocument, DataTypePresent

logger = logging.getLogger(__name__)

class PageIndexer:
    """
    Stage 4 — Page Indexing
    
    Organizes LDUs into a hierarchical section tree (PageIndex).
    """

    def __init__(self):
        pass

    def index(self, doc: ExtractedDocument, ldus: list[LDU]) -> PageIndex:
        """Build a PageIndex tree from LDUs."""
        logger.info("[%s] Indexing %d LDUs …", doc.document_id, len(ldus))
        
        sections: list[SectionNode] = []
        current_section: SectionNode | None = None
        
        # Mapping helpers
        ldu_to_section = {}
        page_to_sections = {}

        for ldu in ldus:
            # If it's a heading, start a new section
            if ldu.ldu_type == "section_heading":
                section_id = f"sec-{ldu.ldu_id}"
                new_section = SectionNode(
                    section_id=section_id,
                    title=ldu.content,
                    level=1, # Default level 1 for now
                    page_references=ldu.page_references,
                    ldu_ids=[ldu.ldu_id]
                )
                sections.append(new_section)
                current_section = new_section
            else:
                # Add LDU to current section or create a default "Intro" section
                if not current_section:
                    current_section = SectionNode(
                        section_id=f"sec-intro-{doc.document_id}",
                        title="Introduction",
                        level=1,
                        page_references=ldu.page_references,
                        ldu_ids=[]
                    )
                    sections.append(current_section)
                
                current_section.ldu_ids.append(ldu.ldu_id)
                # Update section page range
                for p in ldu.page_references:
                    if p not in current_section.page_references:
                        current_section.page_references.append(p)
                
            # Maps
            ldu_to_section[ldu.ldu_id] = current_section.section_id
            for p in ldu.page_references:
                if p not in page_to_sections:
                    page_to_sections[p] = []
                if current_section.section_id not in page_to_sections[p]:
                    page_to_sections[p].append(current_section.section_id)

        # Build Global data types list
        doc_data_types = [DataTypePresent.TEXT]
        if doc.tables: doc_data_types.append(DataTypePresent.TABLE)
        if doc.figures: doc_data_types.append(DataTypePresent.FIGURE)

        index = PageIndex(
            document_id=doc.document_id,
            sections=sections,
            global_data_types=doc_data_types,
            ldu_to_section=ldu_to_section,
            page_to_sections=page_to_sections
        )
        
        logger.info("[%s] PageIndex built with %d sections.", doc.document_id, len(sections))
        return index
