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
        self.max_tokens = MAX_CONTEXT_TOKENS if max_tokens is None else max_tokens
    
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
        context_documents = []
        seen_chunk_ids = set()
        max_results = state.get("context_result_limit", 10)
        max_tokens = state.get("context_max_tokens", self.max_tokens)
        result_items = results if max_results is None else results[: int(max_results)]
        
        for i, paper in enumerate(result_items, 1):
            chunk_text = (
                paper.get("chunk_text")
                or paper.get("text")
                or paper.get("document")
                or paper.get("snippet")
                or paper.get("abstract", "N/A")
            )
            paper_id = paper.get("paper_id", "N/A")
            chunk_id = paper.get("chunk_id") or (
                f"{paper_id}_chunk_{int(paper.get('chunk_index', 0)):04d}"
                if paper.get("chunk_index") not in (None, "N/A", -1)
                else "N/A"
            )
            chunk_index = paper.get("chunk_index", "N/A")
            chunk_start = paper.get("chunk_start", "N/A")
            chunk_length = paper.get("chunk_length", len(chunk_text))
            similarity = paper.get("similarity", paper.get("score", paper.get("rrf_score", 0)))
            source_url = paper.get("source_url", "")

            # Keep only one instance per chunk so the prompt does not waste
            # space on near-duplicate results from different retrieval stages.
            if chunk_id and chunk_id in seen_chunk_ids:
                continue
            if chunk_id:
                seen_chunk_ids.add(chunk_id)

            display_rank = len(context_documents) + 1

            # Format paper section
            paper_section = f"""[{display_rank}] {paper.get('title', 'Unknown Title')}
    Paper ID: {paper_id}
    Chunk ID: {chunk_id}
    Chunk Index: {chunk_index}
    Chunk Start: {chunk_start}
    Chunk Length: {chunk_length}
    Source URL: {source_url}
    Score: {similarity:.4f}
    
    Chunk Text:
    {chunk_text}
"""
            
            # Count tokens (rough estimate: 1 token ≈ 4 characters)
            tokens_in_section = len(paper_section) // 4
            
            if max_tokens is not None and token_count + tokens_in_section > max_tokens:
                print(f"i Reached token limit ({token_count}/{max_tokens}), stopping")
                break
            
            context_parts.append(paper_section)
            token_count += tokens_in_section
            context_documents.append({
                "rank": display_rank,
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "title": paper.get("title"),
                "source_url": source_url,
                "score": paper.get("score", paper.get("rrf_score", 0)),
                "source_score": paper.get("source_score", paper.get("score", paper.get("rrf_score", 0))),
                "chunk_index": paper.get("chunk_index"),
                "chunk_start": paper.get("chunk_start"),
                "chunk_length": paper.get("chunk_length"),
                "chunk_text": chunk_text,
            })
        
        # Format final context
        context_text = f"""# Retrieved Chunks for RAG

Below are the most relevant chunks from our database:

{chr(10).join(context_parts)}

---
Use ONLY the information from the above chunks to answer the user's question.
If information is not in the chunks, say "I couldn't find information about X in the chunks."
Always cite which chunk(s) your answer comes from using the numbered reference labels like [1], [2].
"""
        
        state["context_documents"] = context_documents
        state["context_text"] = context_text
        state["context_size"] = token_count
        state["execution_path"] = state.get("execution_path", []) + ["context_extractor"]
        
        print(f"[OK] Context extracted: {len(context_documents)} papers, {token_count} tokens")
        return state
