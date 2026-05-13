# -*- coding: utf-8 -*-
"""
Agent State Definitions for LangGraph
Using TypedDict instead of Pydantic for LangGraph compatibility
"""

from typing import Optional, List, Literal
from enum import Enum
from typing_extensions import TypedDict


class SearchMode(str, Enum):
    """Available search modes"""
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class IntentType(str, Enum):
    """User intent classification"""
    OOD = "ood"  # Out-of-domain
    GLOBAL = "global"  # In-domain NLP/ML/DL/AI question
    UNCLEAR = "unclear"  # Cannot determine


class AgentState(TypedDict, total=False):
    """Complete state for LangGraph execution - using TypedDict for compatibility"""
    
    # ===== Input =====
    query: str
    session_id: str
    
    # ===== Intent Classification =====
    intent: str
    intent_confidence: float
    intent_explanation: str
    refined_query: str
    
    # ===== Search Execution =====
    search_mode: str
    lexical_results: List[dict]
    semantic_results: List[dict]
    hybrid_results: List[dict]
    reranked_results: List[dict]
    
    # ===== RAG Context =====
    context_documents: List[dict]
    context_text: str
    context_size: int
    
    # ===== Generation =====
    initial_answer: str
    answer_citations: List[dict]
    
    # ===== Evaluation & Feedback =====
    is_answer_good: bool
    answer_confidence: float
    feedback_reason: str
    needs_external_search: bool
    
    # ===== External Data =====
    external_papers: List[dict]
    external_pdfs_parsed: List[dict]
    
    # ===== Final Output =====
    final_answer: str
    final_sources: List[dict]
    final_confidence: float
    execution_path: List[str]
    execution_time_ms: int


def create_initial_state(query: str, session_id: str = "") -> AgentState:
    """Create initial state with all required fields"""
    return {
        "query": query,
        "session_id": session_id,
        "intent": "unclear",
        "intent_confidence": 0.0,
        "intent_explanation": "",
        "refined_query": "",
        "search_mode": "hybrid",
        "lexical_results": [],
        "semantic_results": [],
        "hybrid_results": [],
        "reranked_results": [],
        "context_documents": [],
        "context_text": "",
        "context_size": 0,
        "initial_answer": "",
        "answer_citations": [],
        "is_answer_good": False,
        "answer_confidence": 0.0,
        "feedback_reason": "",
        "needs_external_search": False,
        "external_papers": [],
        "external_pdfs_parsed": [],
        "final_answer": "",
        "final_sources": [],
        "final_confidence": 0.0,
        "external_search_iteration": 0,
        "max_external_iterations": 2,
        "execution_path": [],
        "execution_time_ms": 0
    }
