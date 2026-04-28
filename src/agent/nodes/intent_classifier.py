# -*- coding: utf-8 -*-
"""
Intent Classifier Node
Classify user intent: OOD, GLOBAL, or SPECIFIC
"""

import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.states import IntentType

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class IntentClassifier:
    """Classify user intent (OOD, GLOBAL, SPECIFIC)"""
    
    def __init__(self):
        """Initialize Gemini LLM for intent classification"""
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            api_key=GOOGLE_API_KEY,
            temperature=0.2
        )
        self.ood_keywords = {
            "weather", "joke", "cook", "recipe", "sports", "music", "song",
            "movie", "game", "travel", "restaurant", "price", "drama",
            "stock", "news", "celebrity", "gossip", "diet", "meme", "horoscope"
        }
    
    def __call__(self, state: dict) -> dict:
        """Classify user intent"""
        print(f"\n[CLASSIFY_INTENT] Processing: '{state.get('query', '')[:80]}...'")
        
        query = state.get("query", "")
        query_lower = query.lower()
        execution_path = state.get("execution_path", [])
        
        # Quick OOD detection
        if any(keyword in query_lower for keyword in self.ood_keywords):
            state["intent"] = "ood"
            state["intent_confidence"] = 0.95
            state["intent_explanation"] = "OOD: Non-NLP query detected"
            state["execution_path"] = execution_path + ["classify_intent"]
            print(f"[OK] Intent: {state['intent']} (Confidence: {state['intent_confidence']})")
            return state
        
        # Use Gemini for detailed classification
        prompt = f"""You are an expert NLP Q&A classifier. Classify the user query into one of these categories:

1. **OOD (Out-of-Domain)**: Non-NLP queries (weather, jokes, cooking, etc.)
2. **GLOBAL**: General NLP questions ("What is BERT?", "Latest SOTA rerank models?")
3. **SPECIFIC**: Questions about a specific paper or paper ID

Query: "{query}"

Respond with valid JSON (no markdown):
{{
  "intent": "OOD" | "GLOBAL" | "SPECIFIC",
  "confidence": 0.0-1.0,
  "explanation": "brief reason",
  "refined_query": "improved search query or paper ID"
}}"""
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)
            
            intent_str = result.get("intent", "unclear").lower()
            state["intent"] = intent_str
            state["intent_confidence"] = result.get("confidence", 0.5)
            state["intent_explanation"] = result.get("explanation", "")
            state["refined_query"] = result.get("refined_query", query)
            
        except Exception as e:
            print(f"! Classification error: {e}")
            state["intent"] = "unclear"
            state["intent_confidence"] = 0.5
            state["refined_query"] = query
        
        state["execution_path"] = execution_path + ["classify_intent"]
        print(f"[OK] Intent: {state['intent']} (Confidence: {state['intent_confidence']})")
        return state
