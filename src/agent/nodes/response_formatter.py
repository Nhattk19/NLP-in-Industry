# -*- coding: utf-8 -*-
"""
Response Formatter Node
Format final output to return to user
"""


class ResponseFormatter:
    """Format final response"""

    def _classify_relevance(self, doc: dict) -> str:
        """Classify relevance using rank first, then score as fallback."""
        rank = doc.get("rank")
        if isinstance(rank, int) and rank > 0:
            if rank <= 3:
                return "high"
            if rank <= 7:
                return "medium"
            return "low"

        score = doc.get("score", 0)
        if score is None:
            score = 0
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        if score >= 0.7:
            return "high"
        if score >= 0.3:
            return "medium"
        return "low"
    
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
                "chunk_id": doc.get("chunk_id"),
                "title": doc.get("title"),
                "source_url": doc.get("source_url", ""),
                "chunk_index": doc.get("chunk_index"),
                "chunk_start": doc.get("chunk_start"),
                "chunk_length": doc.get("chunk_length"),
                "score": doc.get("score", 0),
                "source_score": doc.get("source_score", doc.get("score", 0)),
                "relevance": self._classify_relevance(doc)
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
            for paper in external_papers[:10]:  # Top 10 external chunks
                chunk_text = paper.get("chunk_text") or paper.get("text") or paper.get("snippet", "")
                external_papers_formatted.append({
                    "paper_id": paper.get("paper_id", ""),
                    "chunk_id": paper.get("chunk_id", ""),
                    "title": paper.get("title", ""),
                    "source_url": paper.get("source_url", paper.get("url", "")),
                    "source": paper.get("source", "external"),
                    "chunk_index": paper.get("chunk_index"),
                    "chunk_start": paper.get("chunk_start"),
                    "chunk_length": paper.get("chunk_length"),
                    "score": paper.get("score", paper.get("similarity", 0)),
                    "snippet": chunk_text[:200]
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
