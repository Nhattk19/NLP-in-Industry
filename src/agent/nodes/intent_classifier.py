# -*- coding: utf-8 -*-
"""
Intent Classifier Node
Classify user intent: OOD or GLOBAL
"""

import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class IntentClassifier:
    """Classify user intent (OOD, GLOBAL)."""
    
    def __init__(self):
        """Initialize Gemini LLM for intent classification."""
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            api_key=GOOGLE_API_KEY,
            temperature=0.0
        )

    def _build_prompt(self, query: str) -> str:
        """Build a strict classification prompt for the LLM."""
        return f"""You are an intent classification agent for an academic paper RAG system.

Your job is to classify the user's query into exactly one of two labels:

- OOD: the query is outside NLP/ML/DL/AI and related research topics
- GLOBAL: the query is inside NLP/ML/DL/AI and should go through RAG retrieval

Important:
- Treat queries about embeddings, transformers, attention, BERT, GPT, RAG, LLMs,
  machine learning, deep learning, optimization, classification, retrieval,
  recommendation, NLP tasks, and AI research as GLOBAL.
- Treat casual topics like weather, cooking, sports, entertainment, shopping,
  finance news, travel, and similar general-life topics as OOD.
- If the query is ambiguous, prefer GLOBAL so it can be retrieved from the paper DB.

Query: "{query}"

Return ONLY valid JSON with this exact schema:
{{
  "intent": "OOD" | "GLOBAL",
  "confidence": 0.0-1.0,
  "explanation": "short reason",
  "refined_query": "a retrieval-friendly search query"
}}"""

    def _parse_response(self, response_text: str, query: str) -> dict:
        """Parse the model response and normalize the output."""
        text = (response_text or "").strip()
        if not text:
            raise ValueError("Empty classifier response")

        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        result = json.loads(text)
        intent_str = str(result.get("intent", "GLOBAL")).strip().lower()
        if intent_str not in {"ood", "global"}:
            intent_str = "global"

        refined_query = str(result.get("refined_query", query) or query).strip()
        explanation = str(result.get("explanation", "")).strip()

        confidence_raw = result.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        return {
            "intent": intent_str,
            "confidence": confidence,
            "explanation": explanation,
            "refined_query": refined_query,
        }
    
    def __call__(self, state: dict) -> dict:
        """Classify user intent."""
        print(f"\n[CLASSIFY_INTENT] Processing: '{state.get('query', '')[:80]}...'")
        
        query = state.get("query", "")
        execution_path = state.get("execution_path", [])
        
        prompt = self._build_prompt(query)
        
        try:
            response = self.llm.invoke(prompt)
            parsed = self._parse_response(getattr(response, "content", ""), query)

            state["intent"] = parsed["intent"]
            state["intent_confidence"] = parsed["confidence"]
            state["intent_explanation"] = parsed["explanation"]
            state["refined_query"] = parsed["refined_query"]
            
        except Exception as e:
            print(f"! Classification error: {e}")
            # Safe fallback: treat unknown queries as GLOBAL so they still
            # go through RAG retrieval instead of being dropped.
            state["intent"] = "global"
            state["intent_confidence"] = 0.5
            state["intent_explanation"] = "Fallback to GLOBAL on classifier error"
            state["refined_query"] = query
        
        state["execution_path"] = execution_path + ["classify_intent"]
        print(f"[OK] Intent: {state['intent']} (Confidence: {state['intent_confidence']})")
        return state
