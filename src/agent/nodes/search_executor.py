# -*- coding: utf-8 -*-
"""
Search Executor Node
Execute search: Lexical (BM25), Semantic (ChromaDB), or Hybrid (RRF)
Reuses existing code from src/bm25/ and src/chromadb/
"""

import json
from src.agent.states import SearchMode, IntentType
from src.bm25.search_bm25 import BM25Searcher
from src.config import DATA_PATH
from src.search_engine import apply_rrf_merge


class SearchExecutor:
    """Execute search based on selected mode"""
    
    def __init__(self):
        """Initialize search engines"""
        print("[SEARCH_EXECUTOR] Initializing search engines...")
        try:
            self.bm25_searcher = BM25Searcher(DATA_PATH)
            print("[OK] BM25 Searcher initialized")
        except Exception as e:
            print(f"! BM25 Searcher error: {e}")
            self.bm25_searcher = None
        
        # Import ChromaDB retriever
        try:
            from src.chromadb.retrieve import search as chroma_search
            from src.chromadb.retrieve import collection as chroma_collection
            self.chroma_search = chroma_search
            self.chroma_collection = chroma_collection
            print("[OK] ChromaDB Retriever initialized")
        except Exception as e:
            print(f"! ChromaDB error: {e}")
            self.chroma_search = None
            self.chroma_collection = None
    
    def __call__(self, state: dict) -> dict:
        """Execute search"""
        
        # Skip search for OOD queries
        intent = state.get("intent", "unclear")
        if intent == "ood" or intent == IntentType.OOD:
            print("- [SEARCH_EXECUTOR] Skipping search for OOD query")
            state["execution_path"] = state.get("execution_path", []) + ["search_executor"]
            return state
        
        print(f"\n[SEARCH_EXECUTOR] Mode: {state.get('search_mode', 'hybrid')}")
        
        query = state.get("refined_query") or state.get("query", "")
        top_k = 10
        
        search_mode = state.get("search_mode", "hybrid")
        
        if search_mode == "lexical" or search_mode == SearchMode.LEXICAL:
            state = self._lexical_search(state, query, top_k)
        elif search_mode == "semantic" or search_mode == SearchMode.SEMANTIC:
            state = self._semantic_search(state, query, top_k)
        elif search_mode == "hybrid" or search_mode == SearchMode.HYBRID:
            state = self._hybrid_search(state, query, top_k)
        elif search_mode == "paper_specific" or search_mode == SearchMode.PAPER_SPECIFIC:
            state = self._paper_specific_search(state, query, top_k)
        
        state["execution_path"] = state.get("execution_path", []) + ["search_executor"]
        return state
    
    def _lexical_search(self, state: dict, query: str, top_k: int) -> dict:
        """BM25 Lexical Search"""
        if not self.bm25_searcher:
            print("X BM25 Searcher not available")
            return state
        
        try:
            results = self.bm25_searcher.search(query, top_k=top_k)
            state["lexical_results"] = results
            state["reranked_results"] = results
            print(f"[OK] Lexical Search: {len(results)} results")
        except Exception as e:
            print(f"X Lexical search error: {e}")
        
        return state
    
    def _semantic_search(self, state: dict, query: str, top_k: int) -> dict:
        """ChromaDB Semantic Search"""
        if not self.chroma_search:
            print("X ChromaDB not available, falling back to lexical")
            return self._lexical_search(state, query, top_k)
        
        try:
            results = self.chroma_search(query, top_k=top_k)
            state["semantic_results"] = results
            state["reranked_results"] = results
            print(f"[OK] Semantic Search: {len(results)} results")
        except Exception as e:
            print(f"X Semantic search error: {e}")
            return self._lexical_search(state, query, top_k)
        
        return state
    
    def _hybrid_search(self, state: dict, query: str, top_k: int) -> dict:
        """Hybrid RRF (Reciprocal Rank Fusion)"""
        
        lexical_results = []
        semantic_results = []
        
        if self.bm25_searcher:
            try:
                lexical_results = self.bm25_searcher.search(query, top_k=100)
                state["lexical_results"] = lexical_results
            except Exception as e:
                print(f"! Lexical search error: {e}")
        
        if self.chroma_search:
            try:
                semantic_results = self.chroma_search(query, top_k=100)
                state["semantic_results"] = semantic_results
            except Exception as e:
                print(f"! Semantic search error: {e}")
        
        if not lexical_results:
            lexical_results = semantic_results
        if not semantic_results:
            semantic_results = lexical_results
        
        # Use generic RRF merge from search_engine
        try:
            hybrid_results = apply_rrf_merge(lexical_results, semantic_results, top_k)
            state["hybrid_results"] = hybrid_results
            state["reranked_results"] = hybrid_results
            print(f"[OK] Hybrid Search (RRF): {len(hybrid_results)} results")
        except Exception as e:
            print(f"X RRF merge error: {e}")
            state["reranked_results"] = lexical_results or semantic_results
        
        return state
    
    def _paper_specific_search(self, state: dict, query: str, top_k: int) -> dict:
        """Search within a specific paper"""
        paper_id_context = state.get("paper_id_context", "")
        print(f"Paper-specific search for {paper_id_context}")
        
        if self.chroma_search and paper_id_context:
            try:
                results = self.chroma_search(query,top_k=top_k, where={"paper_id": paper_id_context})
                state["semantic_results"] = results
                state["reranked_results"] = results
                print(f"[OK] Paper-specific Search: {len(results)} results")
            except Exception as e:
                print(f"X Paper-specific search error: {e}")
        
        return state
