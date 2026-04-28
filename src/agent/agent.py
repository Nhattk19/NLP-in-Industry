# -*- coding: utf-8 -*-
"""
Main LangGraph Agent
Orchestrates all nodes and manages the RAG flow
Works with dict state (no Pydantic conversion)
"""

import os
import time
import json
from langgraph.graph import StateGraph, END

# Disable ChromaDB telemetry BEFORE any imports
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

from src.agent.states import create_initial_state, IntentType, SearchMode
from src.agent.nodes.intent_classifier import IntentClassifier
from src.agent.nodes.search_executor import SearchExecutor
from src.agent.nodes.context_extractor import ContextExtractor
from src.agent.nodes.answer_generator import AnswerGenerator
from src.agent.nodes.result_evaluator import ResultEvaluator
from src.agent.nodes.external_searcher import ExternalSearcher
from src.agent.nodes.response_formatter import ResponseFormatter


class PaperRAGAgent:
    """Complete RAG Agent for NLP Paper Search and Q&A"""
    
    def __init__(self):
        """Initialize agent components"""
        print("[AGENT] Initializing PaperRAGAgent...")
        
        # Initialize nodes
        self.intent_classifier = IntentClassifier()
        self.search_executor = SearchExecutor()
        self.context_extractor = ContextExtractor()
        self.answer_generator = AnswerGenerator()
        self.result_evaluator = ResultEvaluator()
        self.external_searcher = ExternalSearcher()
        self.response_formatter = ResponseFormatter()
        
        # Build graph
        self.graph = self._build_graph()
        print("[INIT] PaperRAGAgent initialized!")
    
    def _build_graph(self):
        """Build LangGraph state machine with feedback loop for external search"""
        
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("classify_intent", self.intent_classifier)
        workflow.add_node("select_search_mode", self._select_search_mode)
        workflow.add_node("execute_search", self.search_executor)
        workflow.add_node("extract_context", self.context_extractor)
        workflow.add_node("generate_answer", self.answer_generator)
        workflow.add_node("evaluate_result", self.result_evaluator)
        workflow.add_node("external_search", self.external_searcher)
        workflow.add_node("re_search_external", self._re_search_with_external)
        workflow.add_node("re_evaluate_result", self._re_evaluate_after_external)
        workflow.add_node("format_response", self.response_formatter)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Main flow
        workflow.add_edge("classify_intent", "select_search_mode")
        workflow.add_edge("select_search_mode", "execute_search")
        workflow.add_edge("execute_search", "extract_context")
        workflow.add_edge("extract_context", "generate_answer")
        workflow.add_edge("generate_answer", "evaluate_result")
        
        # First evaluation: decide if external search needed
        workflow.add_conditional_edges(
            "evaluate_result",
            self._should_do_external_search,
            {
                "skip": "format_response",
                "external": "external_search",
            }
        )
        
        # External search flow
        workflow.add_edge("external_search", "re_search_external")
        workflow.add_edge("re_search_external", "re_evaluate_result")
        
        # Second evaluation: check if loop needed
        workflow.add_conditional_edges(
            "re_evaluate_result",
            self._should_continue_external_search,
            {
                "stop": "format_response",
                "retry": "external_search",
            }
        )
        
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    def _select_search_mode(self, state: dict) -> dict:
        """Select appropriate search mode based on intent"""
        
        print("\n[SELECT_SEARCH_MODE]")
        
        intent = state.get("intent", "unclear")
        if intent == "ood":
            # No search for OOD
            state["search_mode"] = "hybrid"
            state["execution_path"] = state.get("execution_path", []) + ["select_search_mode"]
            return state
        
        if intent == "specific":
            state["search_mode"] = "paper_specific"
        else:
            # GLOBAL intent - use hybrid by default
            state["search_mode"] = "hybrid"
        
        print(f"[OK] Search mode selected: {state['search_mode']}")
        state["execution_path"] = state.get("execution_path", []) + ["select_search_mode"]
        return state
    
    def _should_do_external_search(self, state: dict) -> str:
        """Decide whether to do external search (feedback loop)"""
        
        # Skip external search for OOD queries
        intent = state.get("intent", "unclear")
        if intent == "ood":
            return "skip"
        
        # Check if evaluation says we need external search
        if state.get("needs_external_search", False):
            print("\n[DECISION] Triggering external search (answer quality low)")
            return "external"
        
        return "skip"
    
    def _re_search_with_external(self, state: dict) -> dict:
        """Re-run search and context extraction with newly ingested external papers"""
        
        print("\n[RE_SEARCH_WITH_EXTERNAL] Re-searching with external papers...")
        
        if not state.get("external_papers"):
            print("  No external papers to use")
            return state
        
        try:
            # Re-run search on the new data
            state = self.search_executor(state)
            print(f"  [OK] Re-search completed with external papers")
            
            # Re-extract context
            state = self.context_extractor(state)
            
            # Re-generate answer with new context
            state = self.answer_generator(state)
            print(f"  [OK] Answer re-generated with external papers")
            
            # Mark that external papers were used
            state["used_external_papers"] = True
            
        except Exception as e:
            print(f"  [WARN] Re-search error: {str(e)}")
            state["used_external_papers"] = False
        
        state["execution_path"] = state.get("execution_path", []) + ["re_search_external"]
        return state
    
    def _re_evaluate_after_external(self, state: dict) -> dict:
        """Re-evaluate answer quality after external search and regeneration"""
        
        print("\n[RE_EVALUATE] Checking answer quality after external search...")
        
        # Increment iteration counter
        state["external_search_iteration"] = state.get("external_search_iteration", 0) + 1
        
        # Re-run evaluator on the regenerated answer
        state = self.result_evaluator(state)
        
        # Store iteration info
        iteration = state["external_search_iteration"]
        max_iter = state.get("max_external_iterations", 2)
        score = state.get("answer_confidence", 0) * 10
        
        print(f"  Iteration {iteration}/{max_iter}: Score = {score:.0f}/10")
        
        state["execution_path"] = state.get("execution_path", []) + ["re_evaluate_result"]
        return state
    
    def _should_continue_external_search(self, state: dict) -> str:
        """Decide whether to continue external search loop or stop"""
        
        score = state.get("answer_confidence", 0) * 10
        iteration = state.get("external_search_iteration", 0)
        max_iterations = state.get("max_external_iterations", 2)
        
        # If answer is good (score >= 7), stop
        if score >= 7:
            print(f"  [OK] Answer quality good (score {score:.0f}/10), stopping loop")
            return "stop"
        
        # If max iterations reached, stop even if answer is bad
        if iteration >= max_iterations:
            print(f"  [WARN] Max iterations ({max_iterations}) reached with score {score:.0f}/10, stopping")
            return "stop"
        
        # Otherwise, continue with another external search
        print(f"  [INFO] Score {score:.0f}/10 < 7, attempting another external search...")
        return "retry"
    
    def run(self, query: str, session_id: str = "") -> dict:
        """Run the agent on a query"""
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Create initial state
        initial_state = create_initial_state(query, session_id)
        
        # Execute graph
        try:
            result_state = self.graph.invoke(initial_state)
            
            # Calculate execution time
            result_state["execution_time_ms"] = int((time.time() - start_time) * 1000)
            
            # Format output
            output = self.response_formatter.format_for_output(result_state)
            
            print(f"\n[OK] Execution completed in {result_state['execution_time_ms']}ms")
            confidence = result_state.get("final_confidence", 0.0)
            print(f"Confidence: {round(confidence * 100)}%")
            
            return output
            
        except Exception as e:
            print(f"\nX Error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
    
    def run_batch(self, queries: list) -> list:
        """Run agent on multiple queries"""
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}]")
            result = self.run(query)
            results.append(result)
        
        return results


# Singleton agent instance
_agent_instance = None


def get_agent() -> PaperRAGAgent:
    """Get or create singleton agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PaperRAGAgent()
    return _agent_instance
