# -*- coding: utf-8 -*-
"""
Answer Generation Node
Generate answers using Gemini 2.5-Flash with RAG context
"""

import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.states import IntentType

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_TEMP = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
MIN_COMPLETE_ANSWER_CHARS = int(os.getenv("MIN_COMPLETE_ANSWER_CHARS", "200"))


class AnswerGenerator:
    """Generate answers using Gemini 2.5-Flash"""
    
    def __init__(self):
        """Initialize Gemini LLM"""
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMP,
            max_output_tokens=LLM_MAX_TOKENS
        )
        
        self.system_prompt = """You are an expert researcher answering questions about academic papers in NLP/ML/DL/AI.
You answer questions about NLP/ML/DL/AI research papers.

INSTRUCTIONS:
1. Answer the question directly first, in simple language a beginner can follow
2. Ground your answer only in the provided papers
3. Use citations only in the exact form [Chunk: arxiv:1810.04805_chunk_0000]
4. Never use numbered citations like [1], [2], [3] or paper citations like [Paper: ...]
5. Prefer concise paraphrasing; use short quotes only when they add real value
6. If the papers do not cover the topic, say so clearly
7. Stay factual and avoid adding claims that are not supported by the chunks
8. Always finish complete sentences and avoid mid-sentence truncation
9. For list queries, provide a short structured list with 1-2 clear sentences per item
10. If multiple chunks overlap, synthesize them into one coherent explanation instead of repeating facts
11. Prioritize correctness and grounding over answer length

STYLE:
- Clear, plain language
- Explain technical terms briefly the first time you use them
- Use structured bullets only when they help readability
- Give concrete examples from the papers when helpful
- Avoid filler, repetition, and academic-sounding padding
- Prefer chunk-level citations over paper-level citations
For explanation questions, follow this order when useful:
1. Give a simple definition
2. Explain how it works
3. Explain why it matters
4. Add examples only if they are supported by the chunks

OUTPUT REQUIREMENTS:
- For factual concept questions, keep the answer concise but complete
- For list/ranking queries, include 2-3 items only if the user asked for them, each with short explanation + citations
- Do not write a paper-style essay unless the user explicitly asks for one
- Do not repeat the same instruction in multiple sentences
- Never truncate or leave incomplete sentences
- Use bullets or numbering only when they improve clarity
"""

    def _extract_finish_reason(self, response) -> str:
        """Best-effort extraction of the model finish reason."""
        metadata = getattr(response, "response_metadata", {}) or {}
        if isinstance(metadata, dict):
            finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason")
            if finish_reason:
                return str(finish_reason).upper()

        additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
        if isinstance(additional_kwargs, dict):
            finish_reason = additional_kwargs.get("finish_reason")
            if finish_reason:
                return str(finish_reason).upper()

        return ""

    def _looks_truncated(self, answer: str, finish_reason: str = "") -> bool:
        """Heuristically detect answers that likely stopped early."""
        stripped = (answer or "").strip()
        if not stripped:
            return True

        if finish_reason in {"MAX_TOKENS", "LENGTH"}:
            return True

        if len(stripped) < MIN_COMPLETE_ANSWER_CHARS:
            # Short answers are allowed for OOD, but not for RAG paper answers.
            return not stripped.endswith((".", "!", "?", ")", "]"))

        # If the answer ends without sentence punctuation, it may have been cut off.
        return not stripped.endswith((".", "!", "?", ")", "]", "\"", "'"))

    def _continue_answer(self, current_answer: str) -> str:
        """Ask the model to continue a likely-truncated answer once."""
        continuation_prompt = f"""The previous answer was cut off before it finished.

Continue the answer from the exact point where it stopped.
Do not repeat the earlier text.
Keep the same grounding and citation style.
Use only exact citations in the form [Chunk: CHUNK_ID].
Do not use numbered citations like [1], [2], [3] or [Paper: ...].
End with complete sentences.

Previous answer:
{current_answer}

Continue now:"""

        response = self.llm.invoke(continuation_prompt)
        continuation = getattr(response, "content", "") or ""
        if continuation.strip():
            return current_answer.rstrip() + "\n" + continuation.lstrip()

        return current_answer
    
    def __call__(self, state: dict) -> dict:
        """Generate answer"""
        
        # For OOD queries, return direct response
        intent = state.get("intent", "unclear")
        if intent == "ood" or intent == IntentType.OOD:
            print("\n[ANSWER_GENERATOR] Generating OOD response...")
            ood_prompt = f"""The user asked a question outside of the NLP/ML/DL/AI paper domain.
User query: "{state.get('query', '')}"

Provide a helpful but brief response, explaining that you specialize in NLP/ML/DL/AI papers.
Keep it under 200 tokens."""
            
            try:
                response = self.llm.invoke(ood_prompt)
                state["initial_answer"] = response.content
                state["answer_confidence"] = 0.5
            except Exception as e:
                print(f"X Error: {e}")
                state["initial_answer"] = (
                    "I specialize in NLP, ML, DL, and AI papers. "
                    "Please ask about topics in that domain."
                )
            
            state["execution_path"] = state.get("execution_path", []) + ["answer_generator"]
            return state
        
        # For in-domain queries, use RAG
        context_text = state.get("context_text", "")
        if not context_text:
            print("! [ANSWER_GENERATOR] No context available, skipping RAG generation")
            state["execution_path"] = state.get("execution_path", []) + ["answer_generator"]
            return state
        
        print("\n[ANSWER_GENERATOR] Generating RAG answer...")
        
        rag_prompt = f"""{self.system_prompt}

# Retrieved Chunk Context
{context_text}

# User Question
{state.get('query', '')}

Answer the question using ONLY the retrieved context.

Requirements:
- Start with a direct answer
- Support important claims with chunk citations
- Use only exact citations in the form [Chunk: CHUNK_ID]
- Never use [1], [2], [3] or [Paper: ...]
- Use one chunk ID per citation; do not combine multiple chunk IDs inside one bracket
- Keep the response clear, concise, and complete
- If information is missing, explicitly say so
- Avoid repeating the same fact in different words

Now generate the answer:"""
        
        try:
            response = self.llm.invoke(rag_prompt)
            answer = getattr(response, "content", "") or ""
            finish_reason = self._extract_finish_reason(response)

            if self._looks_truncated(answer, finish_reason):
                print(
                    f"! [ANSWER_GENERATOR] Answer may be truncated "
                    f"(finish_reason={finish_reason or 'unknown'}, chars={len(answer)})"
                )
                try:
                    answer = self._continue_answer(answer)
                except Exception as continuation_error:
                    print(f"! [ANSWER_GENERATOR] Continuation failed: {continuation_error}")

            answer = self._sanitize_answer(answer)
            state["initial_answer"] = answer
            
            # Extract citations from answer (simple parsing)
            citations = re.findall(r'\[Chunk:\s*([^\]]+)\]', state["initial_answer"])
            parsed_citations = []
            for citation in citations:
                for citation_id in re.split(r"\s*[,;]\s*", citation.strip()):
                    citation_id = citation_id.strip().rstrip(".,")
                    if not citation_id:
                        continue
                    parsed_citations.append(
                        {
                            "paper_id": citation_id.split("_chunk_", 1)[0]
                            if "_chunk_" in citation_id
                            else citation_id,
                            "chunk_id": citation_id,
                        }
                    )
            state["answer_citations"] = parsed_citations
            
            # Simple confidence scoring
            context_docs = state.get("context_documents", [])
            state["answer_confidence"] = min(0.95, 0.7 + len(context_docs) * 0.05)

            print("\n[ANSWER_GENERATOR] Raw answer:")
            print("-" * 80)
            print(state["initial_answer"])
            print("-" * 80)
            print(f"[OK] Answer generated ({len(state['initial_answer'])} chars, {len(citations)} citations)")
            
        except Exception as e:
            print(f"X Generation error: {e}")
            state["initial_answer"] = "I couldn't generate an answer for this query."
            state["answer_confidence"] = 0.0
        
        state["execution_path"] = state.get("execution_path", []) + ["answer_generator"]
        return state

    def _sanitize_answer(self, answer: str) -> str:
        """Remove citation formats that are not allowed in this pipeline."""
        if not answer:
            return answer

        sanitized = re.sub(r"\[(\d+)\]", "", answer)
        sanitized = re.sub(r"\[Paper:[^\]]+\]", "", sanitized)
        return sanitized
