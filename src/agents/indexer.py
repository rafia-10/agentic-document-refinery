import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import yaml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm import get_chat_model
from src.models.schemas import PageIndex, SectionNode, LDU, ExtractedDocument, DataTypePresent

logger = logging.getLogger(__name__)

class PageIndexer:
    """
    Stage 4 — Page Indexing Engine
    
    Organizes LDUs into a hierarchical section tree (PageIndex)
    and generates LLM-powered section summaries.
    """

    def __init__(self, output_dir: str | Path = ".refinery/pageindex/"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM Setup
        self._model = get_chat_model(purpose="summary")
        
        self._summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the following section of a document in 1-2 concise sentences. "
            "Focus on the main topic and key data points. "
            "Section Title: {title}\nContent:\n{content}"
        )

    def index(self, doc: ExtractedDocument, ldus: list[LDU]) -> PageIndex:
        logger.info("[%s] Building Hierarchical PageIndex …", doc.document_id)
        
        # 1. Build Hierarchical Tree
        # We assume LDUs are ordered. We use current_section tracking.
        root_nodes: list[SectionNode] = []
        stack: list[SectionNode] = []
        
        # Map for lookup helpers
        ldu_to_section = {}
        page_to_sections = {}

        for ldu in ldus:
            if ldu.ldu_type == "section_heading":
                # Create a new section
                new_sec = SectionNode(
                    section_id=f"sec-{ldu.ldu_id}",
                    title=ldu.content,
                    level=1, # Simplification: assume all headings at level 1 for now
                    page_references=ldu.page_references,
                    ldu_ids=[ldu.ldu_id]
                )
                
                # Logic to handle nesting (Rule 4 alignment)
                # In a more advanced version, we'd compare levels (e.g., 1.1 vs 1.1.1)
                root_nodes.append(new_sec)
                stack = [new_sec]
            else:
                if not stack:
                    # Default Intro section
                    intro = SectionNode(
                        section_id=f"sec-intro-{doc.document_id}",
                        title="Introduction",
                        level=1,
                        page_references=ldu.page_references,
                        ldu_ids=[]
                    )
                    root_nodes.append(intro)
                    stack = [intro]
                
                curr = stack[-1]
                curr.ldu_ids.append(ldu.ldu_id)
                # Update page references
                for p in ldu.page_references:
                    if p not in curr.page_references:
                        curr.page_references.append(p)
                
            # Maps
            ldu_to_section[ldu.ldu_id] = stack[-1].section_id
            for p in ldu.page_references:
                if p not in page_to_sections:
                    page_to_sections[p] = []
                if stack[-1].section_id not in page_to_sections[p]:
                    page_to_sections[p].append(stack[-1].section_id)

        # 2. Generate LLM Summaries for each section
        # We'll do this for the top-level sections
        for sec in root_nodes:
            sec.summary = self._generate_summary(sec, ldus)

        # 3. Assemble Final Index
        doc_data_types = [DataTypePresent.TEXT]
        if doc.tables: doc_data_types.append(DataTypePresent.TABLE)
        if doc.figures: doc_data_types.append(DataTypePresent.FIGURE)

        index = PageIndex(
            document_id=doc.document_id,
            sections=root_nodes,
            global_data_types=doc_data_types,
            ldu_to_section=ldu_to_section,
            page_to_sections=page_to_sections
        )
        
        # 4. Persistence (JSON Export)
        self._export_index(index)
        
        logger.info("[%s] PageIndex saved to %s", doc.document_id, self._output_dir)
        return index

    def _generate_summary(self, section: SectionNode, ldus: list[LDU]) -> str:
        """Calls LLM to summarize the content of a section."""
        # Collate content for the section
        id_to_ldu = {ldu.ldu_id: ldu for ldu in ldus}
        content_blocks = [id_to_ldu[lid].content for lid in section.ldu_ids if lid in id_to_ldu]
        full_content = "\n\n".join(content_blocks)[:4000] # Cap content
        
        if not full_content.strip():
            return "No content available for summary."
            
        try:
            chain = self._summary_prompt | self._model
            response = chain.invoke({"title": section.title, "content": full_content})
            return response.content.strip() if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error("Failed to generate summary for %s: %s", section.title, e)
            return "Summary generation failed."

    def _export_index(self, index: PageIndex):
        """Saves the PageIndex as a JSON file."""
        file_path = self._output_dir / f"{index.document_id}.json"
        with open(file_path, "w") as f:
            json.dump(index.model_dump(), f, indent=2)
