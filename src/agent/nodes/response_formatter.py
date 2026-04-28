# -*- coding: utf-8 -*-
"""
Response Formatter Node
Format final output to return to user
"""


class ResponseFormatter:
    """Format final response"""
    
    def __call__(self, state: dict) -> dict:
        """Format and prepare final response"""
        
        print("\n[RESPONSE_FORMATTER] Formatting response...")
        
        # Set final answer
        state["final_answer"] = state.get("initial_answer", "")
        
        # Prepare final sources
        final_sources = []
        for doc in state.get("context_documents", []):
            final_sources.append({
                "paper_id": doc.get("paper_id"),
                "title": doc.get("title"),
                "score": doc.get("score", 0),
                "relevance": "high" if doc.get("score", 0) > 0.7 else "medium"
            })
        
        state["final_sources"] = final_sources
        
        # Set final confidence
        state["final_confidence"] = state.get("answer_confidence", 0.0)
        
        state["execution_path"] = state.get("execution_path", []) + ["response_formatter"]
        
        print(f"[OK] Response formatted with {len(final_sources)} sources")
        return state
    
    def format_for_output(self, state: dict) -> dict:
        """Convert state to JSON-serializable output"""
        
        external_papers = state.get("external_papers", [])
        external_papers_formatted = []
        if external_papers:
            for paper in external_papers[:5]:  # Top 5 external papers
                external_papers_formatted.append({
                    "title": paper.get("title", ""),
                    "url": paper.get("url", ""),
                    "source": paper.get("source", "external"),
                    "snippet": paper.get("snippet", "")[:200]
                })
        
        return {
            "success": True,
            "query": state.get("query", ""),
            "intent": state.get("intent", "unclear"),
            "answer": state.get("final_answer", ""),
            "sources": state.get("final_sources", []),
            "confidence": round(state.get("final_confidence", 0.0), 2),
            "search_mode_used": state.get("search_mode", "hybrid"),
            "execution_time_ms": state.get("execution_time_ms", 0),
            "external_search_triggered": state.get("needs_external_search", False),
            "used_external_papers": state.get("used_external_papers", False),
            "external_papers": external_papers_formatted,
            "feedback_info": {
                "answer_is_good": state.get("is_answer_good", False),
                "reason": state.get("feedback_reason", "")
            } if state.get("feedback_reason") else None,
            "execution_path": state.get("execution_path", [])
        }
