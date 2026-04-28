# -*- coding: utf-8 -*-
"""
Context Extraction Node
Extract context from search results for RAG usage
"""

import os
from dotenv import load_dotenv
from src.agent.states import IntentType

load_dotenv()

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))


class ContextExtractor:
    """Extract context from search results"""
    
    def __init__(self, max_tokens: int = None):
        """Initialize context extractor"""
        self.max_tokens = max_tokens or MAX_CONTEXT_TOKENS
    
    def __call__(self, state: dict) -> dict:
        """Extract context from search results"""
        
        # Skip for OOD queries
        intent = state.get("intent", "unclear")
        if intent == "ood" or intent == IntentType.OOD:
            print("- [CONTEXT_EXTRACTOR] Skipping context for OOD query")
            state["execution_path"] = state.get("execution_path", []) + ["context_extractor"]
            return state
        
        print(f"\n[CONTEXT_EXTRACTOR] Building context...")
        
        # Get results based on search mode
        results = state.get("reranked_results") or state.get("lexical_results") or state.get("semantic_results", [])
        
        if not results:
            print("! No results to extract context from")
            state["execution_path"] = state.get("execution_path", []) + ["context_extractor"]
            return state
        
        # Build context
        context_parts = []
        token_count = 0
        context_documents = state.get("context_documents", [])
        
        for i, paper in enumerate(results[:10], 1):
            # Format paper section
            paper_section = f"""[{i}] {paper.get('title', 'Unknown Title')}
    Paper ID: {paper.get('paper_id', 'N/A')}
    Score: {paper.get('score', paper.get('rrf_score', 0)):.4f}
    
    Abstract:
    {paper.get('abstract', 'N/A')[:500]}
"""
            
            # Count tokens (rough estimate: 1 token ≈ 4 characters)
            tokens_in_section = len(paper_section) // 4
            
            if token_count + tokens_in_section > self.max_tokens:
                print(f"i Reached token limit ({token_count}/{self.max_tokens}), stopping")
                break
            
            context_parts.append(paper_section)
            token_count += tokens_in_section
            context_documents.append({
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "score": paper.get("score", paper.get("rrf_score", 0))
            })
        
        # Format final context
        context_text = f"""# Retrieved Papers for RAG

Below are the most relevant papers from our database:

{chr(10).join(context_parts)}

---
Use ONLY the information from the above papers to answer the user's question.
If information is not in the papers, say "I couldn't find information about X in the papers."
Always cite which paper(s) your answer comes from using the Paper ID.
"""
        
        state["context_documents"] = context_documents
        state["context_text"] = context_text
        state["context_size"] = token_count
        state["execution_path"] = state.get("execution_path", []) + ["context_extractor"]
        
        print(f"[OK] Context extracted: {len(context_documents)} papers, {token_count} tokens")
        return state
