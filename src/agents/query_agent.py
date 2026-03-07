import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.utils.llm import get_chat_model
from src.models.schemas import PageIndex, LDU
from src.data.fact_table import FactTableManager
from src.data.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

def get_query_tools():
    """Returns a list of tools for the QueryAgent, with explicit topic-based navigation and smarter routing."""
    data_dir = Path(".refinery/")
    fact_table = FactTableManager(data_dir / "fact_table.db")
    vector_store = VectorStoreManager(data_dir / "vector_store/")
    pageindex_dir = data_dir / "pageindex/"

    @tool
    def pageindex_navigate(document_id: str, topic: str = None) -> str:
        """
        Browse the hierarchical section tree of a specific document.
        Optionally filter by topic or section title for explicit traversal.
        Returns the titles, levels, summaries, and optionally content for matching sections.
        """
        path = pageindex_dir / f"{document_id}.json"
        if not path.exists():
            return f"PageIndex for {document_id} not found."

        with open(path, "r") as f:
            index_data = json.load(f)
            sections = index_data.get("sections", [])

        output = [f"PageIndex for {document_id}:"]
        matched = []
        for sec in sections:
            if topic:
                if topic.lower() in sec["title"].lower() or (sec.get("summary") and topic.lower() in sec["summary"].lower()):
                    matched.append(sec)
            else:
                matched.append(sec)
        if not matched:
            return f"No sections found matching topic '{topic}'."
        for sec in matched:
            output.append(f"- [{sec['level']}] {sec['title']} (Pages {sec['page_references']}): {sec.get('summary', 'No summary.')}")
        return "\n".join(output)

    @tool
    def semantic_search(query: str, document_id: str = None, section_title: str = None, k: int = 5) -> str:
        """
        Search the document corpus using semantic vector search.
        Optionally restrict to a document or section for deterministic topic-based retrieval.
        Returns the most relevant LDUs with their full metadata for citation.
        """
        results = vector_store.search(query, k=k)
        if document_id:
            results = [r for r in results if r["document_id"] == document_id]
        if section_title:
            results = [r for r in results if section_title.lower() in r.get("section_title", "").lower()]
        if not results:
            return "No relevant results found."
        output = ["Semantic Search Results:"]
        for res in results:
            meta = {
                "ldu_id": res["ldu_id"],
                "document_id": res["document_id"],
                "page_references": res["page_references"],
                "content_hash": res["content_hash"],
                "bounding_box": res.get("bounding_box"),
                "section_title": res.get("section_title")
            }
            output.append(
                f"SOURCE_METADATA: {json.dumps(meta)}\n"
                f"Content: {res['content']}\n---"
            )
        return "\n".join(output)

    @tool
    def structured_query(sql_query: str, document_id: str = None, section_id: str = None) -> str:
        """
        Query the SQLite FactTable for numerical or tabular data.
        Optionally restrict to a document or section for deterministic retrieval.
        The 'facts' table has: [document_id, page_number, fact_type, entity, value, unit, context, source_ldu_id].
        The 'tables' table has: [table_id, document_id, headers, data].
        """
        if document_id:
            sql_query = sql_query.strip()
            if sql_query.lower().startswith("select") and "where" not in sql_query.lower():
                sql_query += f" WHERE document_id = '{document_id}'"
        results = fact_table.query_facts(sql_query)
        if not results:
            return "No records found matching the query."
        return f"Structured Query Results:\n{json.dumps(results)}"

    return [pageindex_navigate, semantic_search, structured_query]

# --- LangGraph Agent ---

class AgentState(dict):
    messages: List[BaseMessage]

class QueryAgent:
    """
    LangGraph agent for sophisticated document querying and claim verification.
    """

    def __init__(self):
        self._tool_list = get_query_tools()
        
        # LLM Setup
        self._model = get_chat_model().bind_tools(self._tool_list)

        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self._tool_list))

        # Edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _call_model(self, state: AgentState):
        response = self._model.invoke(state["messages"])
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "continue"
        return "end"

    def ask(self, question: str, audit_mode: bool = False) -> Any:
        """
        Execute the agent to answer a question. 
        In audit_mode, it returns a dict with 'answer' and 'provenance'.
        """
        from src.models.schemas import ProvenanceChain, ProvenanceRecord
        import hashlib
        
        prompt = question
        if audit_mode:
            prompt += (
                "\n\nAUDIT MODE: You must verify every claim with a source citation. "
                "For every fact you state, you MUST find the corresponding SOURCE_METADATA "
                "from the tools and include it in your final response as a JSON array of citations "
                "at the very end of your message, wrapped in <provenance_links> tags."
            )
            
        inputs = {"messages": [HumanMessage(content=prompt)]}
        result = self._graph.invoke(inputs)
        content = result["messages"][-1].content
        
        if not audit_mode:
            return content

        # Extract provenance links from LLM output
        import re
        try:
            prov_match = re.search(r'<provenance_links>(.*?)</provenance_links>', content, re.DOTALL)
            records = []
            if prov_match:
                prov_data = json.loads(prov_match.group(1))
                for item in prov_data:
                    # Item should have document_id, page_references [0], bounding_box, content_hash
                    records.append(ProvenanceRecord(
                        document_name=item.get("document_id", "Unknown"),
                        document_id=item.get("document_id", "Unknown"),
                        page_number=item.get("page_references", [1])[0],
                        bounding_box=item.get("bounding_box"),
                        content_hash=item.get("content_hash", ""),
                        excerpt=item.get("excerpt")
                    ))
            
            chain = ProvenanceChain(
                chain_id=hashlib.sha256(content.encode()).hexdigest()[:8],
                derived_artefact_id=hashlib.sha256(question.encode()).hexdigest()[:8],
                records=records if records else [ProvenanceRecord(
                    document_name="Unknown", document_id="Unknown", page_number=1, content_hash="", excerpt="No citations found."
                )]
            )
            
            # Remove the raw provenance tags from the user-facing answer
            clean_answer = re.sub(r'<provenance_links>.*?</provenance_links>', '', content, flags=re.DOTALL).strip()
            
            return {
                "answer": clean_answer,
                "provenance": chain
            }
        except Exception as e:
            logger.error("Failed to parse provenance: %s", e)
            return {
                "answer": content,
                "provenance": None
            }
