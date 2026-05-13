# -*- coding: utf-8 -*-
"""
Result Evaluator Node
Evaluate answer quality (0-10 scoring)
Decide whether external search is needed
"""

import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.states import IntentType

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EXTERNAL_SEARCH_CONFIDENCE_THRESHOLD = 0.7

class ResultEvaluator:
    """Evaluate answer quality"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
    
    def __call__(self, state: dict) -> dict:
        """Evaluate result quality"""
        
        # OOD queries don't need evaluation
        intent = state.get("intent", "unclear")
        if intent == "ood" or intent == IntentType.OOD:
            state["is_answer_good"] = True
            state["answer_confidence"] = 0.9
            state["execution_path"] = state.get("execution_path", []) + ["result_evaluator"]
            return state
        
        print("\n[RESULT_EVALUATOR] Evaluating answer quality...")
        
        initial_answer = state.get("initial_answer", "")
        if not initial_answer:
            print("X No answer to evaluate")
            state["is_answer_good"] = False
            state["answer_confidence"] = 0.0
            state["needs_external_search"] = True
            state["feedback_reason"] = "Empty answer"
            state["execution_path"] = state.get("execution_path", []) + ["result_evaluator"]
            return state
        
        # Build evaluation prompt
        answer_citations = state.get("answer_citations", [])
        citation_ids = []
        for citation in answer_citations:
            chunk_id = citation.get("chunk_id") or citation.get("paper_id")
            if chunk_id:
                citation_ids.append(chunk_id)
        eval_prompt = f"""Evaluate this Q&A pair on a scale of 0-10:

**Query:** {state.get('query', '')}

**Answer:** {initial_answer}

**Cited Chunks/Papers:** {citation_ids}

**Scoring Criteria:**
- Relevance (40%): Does answer directly address query?
- Grounding (30%): Are claims properly cited at chunk level?
- Completeness (20%): Does answer feel complete?
- Freshness (10%): If SOTA, are papers recent?
- Only request external search if the answer is materially insufficient or grounding is weak.
- Do NOT request external search just because the answer could be a little richer or more detailed.

Respond in JSON:
{{
  "score": 0-10,
  "reason": "Why this score?",
  "issues": ["Issue 1", "Issue 2"],
  "needs_external_search": true/false
}}"""
        
        try:
            response = self.llm.invoke(eval_prompt)
            response_text = response.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            if not response_text:
                raise ValueError("Empty response from evaluator")
            
            evaluation = json.loads(response_text)
            
            score = evaluation.get("score", 5)
            state["is_answer_good"] = score >= 7
            state["answer_confidence"] = score / 10.0
            state["feedback_reason"] = evaluation.get("reason", "")
            model_requests_external = bool(evaluation.get("needs_external_search", False))
            state["needs_external_search"] = (
                score < 6
                and state["answer_confidence"] < EXTERNAL_SEARCH_CONFIDENCE_THRESHOLD
            )

            if model_requests_external and not state["needs_external_search"]:
                print("! External search request overridden: answer score is already good enough")

            print(f"[OK] Evaluation Score: {score}/10 - {evaluation.get('reason')}")
            
            if state["needs_external_search"]:
                print(f"! External search triggered: {evaluation.get('reason')}")
            
        except json.JSONDecodeError as e:
            print(f"! JSON parse error in evaluation: {e}")
            print(f"  Response was: {response.content[:100]}...")
            # Default: assume answer is mediocre, trigger external search
            answer_len = len(initial_answer)
            state["is_answer_good"] = answer_len > 200
            state["answer_confidence"] = min(0.6, answer_len / 500)
            state["needs_external_search"] = (
                answer_len < 300
                and state["answer_confidence"] < EXTERNAL_SEARCH_CONFIDENCE_THRESHOLD
            )
            state["feedback_reason"] = "Evaluation unavailable, checking answer length"
            print(f"! Falling back: answer_good={state['is_answer_good']}, trigger_external={state['needs_external_search']}")
            
        except Exception as e:
            print(f"! Evaluation error: {e}, assuming mediocre answer")
            answer_len = len(initial_answer)
            state["is_answer_good"] = answer_len > 200
            state["answer_confidence"] = 0.6
            state["needs_external_search"] = (
                answer_len < 300
                and state["answer_confidence"] < EXTERNAL_SEARCH_CONFIDENCE_THRESHOLD
            )
            state["feedback_reason"] = f"Evaluation failed: {str(e)}"
        
        state["execution_path"] = state.get("execution_path", []) + ["result_evaluator"]
        return state
