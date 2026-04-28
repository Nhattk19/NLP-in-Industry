# -*- coding: utf-8 -*-
"""
Answer Generation Node
Generate answers using Gemini 2.5-Flash with RAG context
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.states import IntentType

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_TEMP = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))


class AnswerGenerator:
    """Generate answers using Gemini 2.5-Flash"""
    
    def __init__(self):
        """Initialize Gemini LLM"""
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMP,
            max_tokens=LLM_MAX_TOKENS
        )
        
        self.system_prompt = """You are an expert NLP researcher answering questions about academic papers.

INSTRUCTIONS:
1. GROUND YOUR ANSWER in the provided papers
2. USE CITATIONS: Reference paper IDs like [Paper: arxiv:1810.04805]
3. BE PRECISE: Quote relevant sections when needed
4. ACKNOWLEDGE LIMITATIONS: If papers don't cover the topic, say so
5. MAINTAIN OBJECTIVITY: Present facts from papers, not your interpretation
6. ALWAYS COMPLETE: Provide full, complete answers - DO NOT cut off mid-sentence
7. BE THOROUGH: Include details, examples, and context from papers
8. For LIST QUERIES: Provide structured list with 2-3 sentences per item

STYLE:
- Clear, comprehensive language
- Technical accuracy
- Structured lists/bullets for multiple items
- Relevant examples from papers
- Always finish sentences completely
- Write SUBSTANTIAL answers (800+ chars for list queries)

OUTPUT REQUIREMENTS:
- MINIMUM 800 characters for list/ranking queries
- COMPLETE, full-length responses (not abbreviated)
- Never truncate or leave incomplete sentences
- Include multiple papers/sources
- Each list item: 2-3 sentences + citations
- Use bullet points or numbering for clarity
"""
    
    def __call__(self, state: dict) -> dict:
        """Generate answer"""
        
        # For OOD queries, return direct response
        intent = state.get("intent", "unclear")
        if intent == "ood" or intent == IntentType.OOD:
            print("\n[ANSWER_GENERATOR] Generating OOD response...")
            ood_prompt = f"""The user asked a question outside of NLP/ML papers domain.
User query: "{state.get('query', '')}"

Provide a helpful but brief response, explaining that you specialize in NLP papers.
Keep it under 200 tokens."""
            
            try:
                response = self.llm.invoke(ood_prompt)
                state["initial_answer"] = response.content
                state["answer_confidence"] = 0.5
            except Exception as e:
                print(f"X Error: {e}")
                state["initial_answer"] = "I specialize in NLP and ML papers. Please ask about NLP-related topics."
            
            state["execution_path"] = state.get("execution_path", []) + ["answer_generator"]
            return state
        
        # For NLP queries, use RAG
        context_text = state.get("context_text", "")
        if not context_text:
            print("! [ANSWER_GENERATOR] No context available, skipping RAG generation")
            state["execution_path"] = state.get("execution_path", []) + ["answer_generator"]
            return state
        
        print("\n[ANSWER_GENERATOR] Generating RAG answer...")
        
        rag_prompt = f"""{self.system_prompt}

# Retrieved Papers Context
{context_text}

# User Question
{state.get('query', '')}

CRITICAL INSTRUCTIONS FOR THIS QUERY:
1. Analyze ALL the papers above carefully
2. Extract SPECIFIC model names, techniques, metrics, and results
3. Write a SUBSTANTIVE answer with AT LEAST 800+ characters
4. For list/ranking queries: provide structured list with explanations
5. Include paper citations: [Paper: ID]
6. DO NOT summarize vaguely - provide DETAILS
7. DO NOT cut off - write COMPLETE sentences and thoughts
8. If query asks for "Top N", list ALL N items with details
9. Compare approaches where applicable
10. Include performance metrics/benchmarks when available

Now generate a DETAILED, COMPREHENSIVE, COMPLETE answer (minimum 800 chars) based ONLY on the retrieved papers. Write your answer now:"""
        
        try:
            response = self.llm.invoke(rag_prompt)
            state["initial_answer"] = response.content
            
            # Extract citations from answer (simple parsing)
            import re
            citations = re.findall(r'\[Paper:\s*([^\]]+)\]', state["initial_answer"])
            state["answer_citations"] = [{"paper_id": c.strip()} for c in citations]
            
            # Simple confidence scoring
            context_docs = state.get("context_documents", [])
            state["answer_confidence"] = min(0.95, 0.7 + len(context_docs) * 0.05)
            
            print(f"[OK] Answer generated ({len(state['initial_answer'])} chars, {len(citations)} citations)")
            
        except Exception as e:
            print(f"X Generation error: {e}")
            state["initial_answer"] = "I couldn't generate an answer for this query."
            state["answer_confidence"] = 0.0
        
        state["execution_path"] = state.get("execution_path", []) + ["answer_generator"]
        return state
