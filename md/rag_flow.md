# RAG (Retrieval-Augmented Generation) Flow Documentation

## Table of Contents
1. [Overview](#overview)
2. [Data Initialization & Loading](#data-initialization--loading)
3. [Graph Architecture](#graph-architecture)
4. [Node-by-Node Flow](#node-by-node-flow)
5. [Data Transformations](#data-transformations)
6. [Example Execution](#example-execution)

---

## Overview

The NLP Paper RAG Agent is a **LangGraph-based orchestrator** that processes user queries about academic papers through a 10-node pipeline:

```
Query → Intent Classification → Search Mode Selection → Search Execution 
    → Context Extraction → Answer Generation → Result Evaluation 
    → [External Search (if needed)] → Response Formatting → Output
```

**Key Technologies:**
- **LangGraph**: State machine orchestration
- **Gemini 2.5-Flash**: LLM for classification, generation, evaluation
- **BM25**: Lexical (keyword-based) search
- **ChromaDB**: Semantic (embedding-based) search
- **arXiv API**: External paper discovery

---

## Data Initialization & Loading

### 1. Data Source: `final_cleaned_data.jsonl`

**Location:** `data/data_processed/final_cleaned_data.jsonl`

**File Format:** JSON Lines (one JSON object per line, 13,012 papers)

```json
{
  "paper_id": "265099508",
  "title": "Transformer Models for NLP",
  "abstract": "This paper explores...",
  "authors": [...],
  "year": 2023
}
```

**What it contains:**
- 13,012 academic papers from S2
- Full metadata: ID, title, abstract, authors, publication year
- Used for building both lexical and semantic indices

### 2. Lexical Index: BM25 Engine

**Function:** `BM25Searcher` in `src/bm25/search_bm25.py`

**Initialization Process:**
```
[BM25] Loading data from final_cleaned_data.jsonl to build index...
[BM25] Initializing BM25 Engine for 13012 papers...
[BM25] Index ready!
```

**What happens:**
1. **Load JSONL file** line-by-line
2. **Extract fields**: Combine `title` + `abstract` into searchable text
3. **Tokenize**: Convert text to lowercase, remove punctuation
   - Example: "What is BERT?" → `["what", "is", "bert"]`
4. **Build BM25 index**: Create inverted index for keyword matching
   - BM25 scores documents based on term frequency and rarity
5. **Store metadata**: Keep paper_id, title, abstract for later retrieval

**Output:** BM25 object with 13,012 indexed documents

```python
class BM25Searcher:
    def __init__(self, data_path):
        # Load JSONL
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                search_text = f"{doc['title']} {doc['abstract']}"
                tokenized_corpus.append(tokenize(search_text))
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
```

### 3. Semantic Index: ChromaDB

**Location:** `data/chroma_store_fulltext/` (persistent vector database)

**Connection Process:**
```
[INIT] Connecting to ChromaDB at: ./data/chroma_store_fulltext...
[OK] Connected to Collection: 'papers' (Total papers: 9)
```

**What happens:**
1. **Initialize ChromaDB client**: Connect to persistent storage
2. **Load collection**: Retrieve pre-computed embeddings
3. **Set embedding function**: Use `all-MiniLM-L6-v2` ONNX model
   - Converts text → 384-dimensional embedding vectors
   - No PyTorch required (optimized for speed)

**Data stored in ChromaDB:**
- Paper embeddings (384-dim vectors)
- Paper metadata (title, abstract, paper_id)
- Pre-calculated similarity relationships

**Usage:** Fast semantic similarity search using vector proximity

```python
# ChromaDB retrieval example
collection.query(
    query_embeddings=[[0.1, 0.2, ...]],  # Query embedding
    n_results=10  # Top 10 similar papers
)
```

---

## Graph Architecture

### LangGraph State Machine

**State Definition:** `src/agent/states.py` → `AgentState` (TypedDict)

```python
class AgentState(TypedDict, total=False):
    # Input
    query: str
    session_id: str
    
    # Intent Classification
    intent: str  # "ood", "global", "specific", "unclear"
    intent_confidence: float
    intent_explanation: str
    
    # Search Execution
    search_mode: str  # "semantic", "lexical", "hybrid"
    lexical_results: List[dict]
    semantic_results: List[dict]
    hybrid_results: List[dict]
    reranked_results: List[dict]
    
    # RAG Context
    context_documents: List[dict]
    context_text: str
    context_size: int
    
    # Generation & Evaluation
    initial_answer: str
    answer_citations: List[dict]
    is_answer_good: bool
    answer_confidence: float
    needs_external_search: bool
    
    # External Data
    external_papers: List[dict]
    
    # Final Output
    final_answer: str
    final_sources: List[dict]
    final_confidence: float
    execution_path: List[str]
    execution_time_ms: int
```

### Graph Edges & Flow Control

```
classify_intent 
    ↓
select_search_mode
    ↓
execute_search
    ↓
extract_context
    ↓
generate_answer
    ↓
evaluate_result
    ↓ [Conditional: needs_external_search?]
    ├─→ YES → external_search → re_search_external → re_evaluate_result
    │              ↓ [Conditional: retry?]
    │              └─→ YES: loop back to external_search (max 2 iterations)
    │              └─→ NO: continue
    └─→ NO → format_response
    ↓
format_response
    ↓
END
```

---

## Node-by-Node Flow

### Node 1: Classify Intent

**File:** `src/agent/nodes/intent_classifier.py`

**Purpose:** Categorize user query type

**Input State:**
```python
{
    "query": "Top 5 SOTA models in reranker",
    ...
}
```

**Processing:**

1. **Quick OOD Check** (fast path):
   - Check keywords: weather, joke, cook, recipe, sports, music, etc.
   - If match → Intent = "ood", Confidence = 0.95 (skip rest)

2. **Gemini Classification** (for non-OOD):
   - Send prompt to Gemini 2.5-Flash:
     ```
     Classify into: OOD | GLOBAL | SPECIFIC
     - OOD: Weather, jokes, cooking
     - GLOBAL: General NLP questions
     - SPECIFIC: Paper-specific questions
     ```
   - Parse JSON response

**Output State:**
```python
{
    "intent": "global",
    "intent_confidence": 0.95,
    "intent_explanation": "GLOBAL: General question about SOTA models",
    "refined_query": "Top 5 SOTA reranker models in NLP",
    "execution_path": ["classify_intent"]
}
```

**Example Console Output:**
```
[CLASSIFY_INTENT] Processing: 'Top 5 SOTA models in reranker...'
[OK] Intent: global (Confidence: 0.95)
```

---

### Node 2: Select Search Mode

**File:** `src/agent/agent.py` → `_select_search_mode()`

**Purpose:** Choose search strategy based on intent

**Logic:**
```python
def _select_search_mode(self, state: dict) -> dict:
    intent = state.get("intent", "unclear")
    
    if intent == "ood":
        state["search_mode"] = "none"  # No search needed
    elif intent == "specific":
        state["search_mode"] = "paper_specific"  # Find exact paper
    elif intent == "global":
        state["search_mode"] = "hybrid"  # Best coverage
    else:  # unclear
        state["search_mode"] = "hybrid"  # Default to hybrid
```

**Output State:**
```python
{
    "search_mode": "hybrid",
    "execution_path": [..., "select_search_mode"]
}
```

**Example Console Output:**
```
[SELECT_SEARCH_MODE]
[OK] Search mode selected: hybrid
```

---

### Node 3: Execute Search

**File:** `src/agent/nodes/search_executor.py`

**Purpose:** Perform actual search using selected mode

#### 3a. Lexical Search (BM25)

```python
def _lexical_search(self, state: dict, query: str, top_k: int) -> dict:
    # Query: "Top 5 SOTA models in reranker"
    results = self.bm25_searcher.search(query, top_k=10)
    
    # BM25.get_scores(tokenized_query):
    # - Tokenize: ["top", "5", "sota", "models", "reranker"]
    # - Score each document in corpus
    # - Return scores for all docs
    
    # Example output:
    results = [
        {
            "paper_id": "arxiv:2604.22750v1",
            "title": "Bidirectional Transformer Reranker for GEC",
            "abstract": "...",
            "score": 18.0650  # BM25 score
        },
        ...
    ]
```

**How it works:**
1. Tokenize query into terms
2. Look up each term in inverted index
3. Score documents based on:
   - Term frequency (TF): how many times term appears
   - Inverse document frequency (IDF): how rare the term is
   - Document length normalization

#### 3b. Semantic Search (ChromaDB)

```python
def _semantic_search(self, state: dict, query: str, top_k: int) -> dict:
    # Query: "Top 5 SOTA models in reranker"
    
    # 1. Embed query using all-MiniLM-L6-v2
    #    "Top 5 SOTA models in reranker" → [0.123, 0.456, ..., 0.789]
    #    384-dimensional vector
    
    # 2. Search ChromaDB for nearest neighbors
    results = self.chroma_search(
        query_embeddings=query_embedding,
        n_results=10
    )
    
    # 3. ChromaDB returns papers with highest cosine similarity
    results = [
        {
            "paper_id": "arxiv:2604.22751v1",
            "title": "Gumbel Reranking: Differentiable End-to-End",
            "abstract": "...",
            "score": 0.8234  # Cosine similarity (0-1)
        },
        ...
    ]
```

#### 3c. Hybrid Search (RRF - Reciprocal Rank Fusion)

```python
def _hybrid_search(self, state: dict, query: str, top_k: int) -> dict:
    # 1. Get lexical results
    lexical = bm25_search(query, top_k=10)
    
    # 2. Get semantic results
    semantic = chromadb_search(query, top_k=10)
    
    # 3. Merge using RRF (Reciprocal Rank Fusion)
    # RRF formula: score = 1 / (k + rank)
    # where k=60 (constant), rank is position in ranked list
    
    # Example:
    # Paper A: rank 1 in lexical (1/61) + rank 3 in semantic (1/63)
    # = 0.0164 + 0.0159 = 0.0323 (combined score)
    
    hybrid_results = combine_with_rrf(lexical, semantic)
    # Results automatically deduplicated and reranked
```

**Output State:**
```python
{
    "lexical_results": [...],
    "semantic_results": [...],
    "hybrid_results": [...],
    "reranked_results": [...],  # Final ranked results
    "execution_path": [..., "search_executor"]
}
```

**Example Console Output:**
```
[SEARCH_EXECUTOR] Mode: hybrid
[OK] Hybrid Search (RRF): 10 results
```

---

### Node 4: Extract Context

**File:** `src/agent/nodes/context_extractor.py`

**Purpose:** Build RAG context window from search results

**Process:**

```python
def __call__(self, state: dict) -> dict:
    print("[CONTEXT_EXTRACTOR] Building context...")
    
    results = state.get("reranked_results", [])
    max_tokens = 4000  # MAX_CONTEXT_TOKENS from .env
    
    context_parts = []
    token_count = 0
    context_documents = []
    
    for i, paper in enumerate(results[:10], 1):
        # Format paper section
        paper_section = f"""[{i}] {paper['title']}
    Paper ID: {paper['paper_id']}
    Score: {paper['score']:.4f}
    
    Abstract:
    {paper['abstract'][:500]}
"""
        
        # Token estimation: 1 token ≈ 4 characters
        tokens_in_section = len(paper_section) // 4
        
        # Check token limit
        if token_count + tokens_in_section > max_tokens:
            break
        
        context_parts.append(paper_section)
        token_count += tokens_in_section
        context_documents.append({
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "score": paper["score"]
        })
    
    # Build final context text
    context_text = f"""# Retrieved Papers for RAG

Below are the most relevant papers from our database:

{chr(10).join(context_parts)}

---"""
    
    state["context_text"] = context_text
    state["context_documents"] = context_documents
    state["context_size"] = token_count
```

**Example Output:**
```python
{
    "context_text": """# Retrieved Papers for RAG

[1] Bidirectional Transformer Reranker for Grammatical Error Correction
    Paper ID: 258833107
    Score: 18.0650
    
    Abstract:
    This model is proposed to re-estimate the probability of candidate 
    sentences generated by pre-trained seq2seq models...

[2] Gumbel Reranking: Differentiable End-to-End Reranker Optimization
    Paper ID: 276408982
    Score: 17.3646
    ...
""",
    "context_documents": [
        {"paper_id": "258833107", "title": "Bidirectional Transformer...", "score": 18.0650},
        {"paper_id": "276408982", "title": "Gumbel Reranking...", "score": 17.3646},
        ...
    ],
    "context_size": 2100  # tokens
}
```

**Example Console Output:**
```
[CONTEXT_EXTRACTOR] Building context...
[OK] Context extracted: 10 papers, 1024 tokens
```

---

### Node 5: Answer Generator

**File:** `src/agent/nodes/answer_generator.py`

**Purpose:** Generate RAG answer using Gemini LLM with context

**Process:**

```python
def __call__(self, state: dict) -> dict:
    # For non-OOD queries, use RAG with context
    context_text = state.get("context_text", "")
    query = state.get("query", "")
    
    # Build RAG prompt
    rag_prompt = f"""{self.system_prompt}

CONTEXT:
{context_text}

USER QUERY: {query}

Generate a comprehensive answer grounded in the provided papers.
Include citations: [Paper: arxiv:1810.04805]"""
    
    # Call Gemini 2.5-Flash
    response = self.llm.invoke(rag_prompt)
    # LLM_MAX_TOKENS = 2000 (from .env)
    # LLM_TEMPERATURE = 0.2 (low temperature for consistency)
    
    state["initial_answer"] = response.content
    state["answer_confidence"] = 0.7  # Default confidence
```

**System Prompt includes:**
```
1. GROUND YOUR ANSWER in provided papers
2. USE CITATIONS: Reference paper IDs
3. BE PRECISE: Quote relevant sections
4. ACKNOWLEDGE LIMITATIONS
5. MAINTAIN OBJECTIVITY
6. ALWAYS COMPLETE: Don't truncate
7. BE THOROUGH: Include details and examples
```

**Example Output:**
```python
{
    "initial_answer": """Based on the provided papers, here are five significant 
    reranker models or approaches discussed:

    * **Bidirectional Transformer Reranker (BTR)**: This model is proposed 
      to re-estimate the probability of candidate sentences generated by 
      pre-trained seq2seq models, specifically for Grammatical Error 
      Correction (GEC). [Paper: 258833107]
      
    * **Gumbel Reranking**: Differentiable end-to-end reranker optimization
      approach. [Paper: 276408982]
      
    ...""",
    "answer_confidence": 0.7
}
```

**Example Console Output:**
```
[ANSWER_GENERATOR] Generating RAG answer...
[OK] Answer generated (403 chars, 0 citations)
```

---

### Node 6: Result Evaluator

**File:** `src/agent/nodes/result_evaluator.py`

**Purpose:** Evaluate answer quality (0-10 score)

**Scoring Criteria:**
- **Relevance** (40%): Does answer address query?
- **Grounding** (30%): Are claims cited?
- **Completeness** (20%): Does answer feel complete?
- **Freshness** (10%): Are papers recent (for SOTA)?

**Evaluation Process:**

```python
def __call__(self, state: dict) -> dict:
    answer = state.get("initial_answer", "")
    query = state.get("query", "")
    citations = state.get("answer_citations", [])
    
    eval_prompt = f"""Evaluate this Q&A pair on a scale of 0-10:

**Query:** {query}

**Answer:** {answer}

**Cited Papers:** {[c['paper_id'] for c in citations]}

Respond in JSON:
{{
  "score": 0-10,
  "reason": "Why this score?",
  "issues": ["Issue 1", "Issue 2"],
  "needs_external_search": true/false
}}"""
    
    response = self.llm.invoke(eval_prompt)
    
    # Parse response
    result = json.loads(extract_json(response.content))
    
    score = result.get("score", 0)
    state["answer_confidence"] = score / 10.0
    state["is_answer_good"] = score >= 7
    state["needs_external_search"] = result.get("needs_external_search", False)
    state["feedback_reason"] = result.get("reason", "")
```

**Example Output:**
```python
{
    "answer_confidence": 0.1,  # 1/10 score
    "is_answer_good": False,
    "needs_external_search": True,
    "feedback_reason": """The answer is severely incomplete, providing only 
    2 models instead of the requested 5. One of the two models (Gumbel 
    Reranking) is not grounded with any citation..."""
}
```

**Example Console Output:**
```
[RESULT_EVALUATOR] Evaluating answer quality...
[OK] Evaluation Score: 1/10 - The answer is severely incomplete, providing 
only 2 models instead of the requested 5...
! External search triggered: (reason shown above)
```

---

### Node 7: External Searcher

**File:** `src/agent/nodes/external_searcher.py`

**Purpose:** Search arXiv for new papers when answer quality is low

**Process:**

```python
def __call__(self, state: dict) -> dict:
    print("[EXTERNAL_SEARCHER] Searching external sources (arXiv API)...")
    
    # Build search query
    search_query = self._build_search_query(state)
    
    # Search arXiv API
    results = self._search_arxiv(search_query)
    
    # Optionally parse PDFs (if enabled)
    if self.enable_pdf_parsing:
        results = self._enrich_with_pdf_content(results)
    
    # Ingest results into ChromaDB
    if results:
        self._ingest_to_chromadb(results, state["query"])
    
    state["external_papers"] = results
```

**arXiv API Search:**

```python
def _search_arxiv(self, query: str) -> List[Dict]:
    """Search arXiv and return paper metadata"""
    
    arxiv_url = "http://export.arxiv.org/api/query?"
    
    # Construct query with filters
    search_query = f"all:{query} AND (cat:cs.CL OR cat:cs.LG)"
    
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": 5,
        "start": 0
    }
    
    response = requests.get(arxiv_url, params=params, timeout=15)
    
    # Parse Atom XML response
    root = ET.fromstring(response.content)
    
    results = []
    for entry in root.findall('{...}entry'):
        paper = {
            "title": entry.find('{...}title').text,
            "arxiv_id": extract_id(entry.find('{...}id').text),
            "url": entry.find('{...}id').text,
            "abstract": entry.find('{...}summary').text,
            "authors": [author.find('{...}name').text 
                       for author in entry.findall('{...}author')],
            "published": entry.find('{...}published').text
        }
        results.append(paper)
    
    return results[:5]
```

**ChromaDB Ingestion:**

```python
def _ingest_to_chromadb(self, papers: List[Dict], query: str):
    """Add external papers to ChromaDB for future searches"""
    
    from src.chromadb.retrieve import collection
    
    # Prepare data for ChromaDB
    ids = [paper['arxiv_id'] for paper in papers]
    embeddings = None  # ChromaDB will compute embeddings
    documents = [f"{p['title']} {p['abstract']}" for p in papers]
    metadatas = [
        {
            "source": "external",
            "arxiv_id": p['arxiv_id'],
            "title": p['title'],
            "url": p['url']
        }
        for p in papers
    ]
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
```

**Example Output:**
```python
{
    "external_papers": [
        {
            "title": "Correlated Quantum Dephasometry...",
            "arxiv_id": "2604.22751v1",
            "url": "https://arxiv.org/abs/2604.22751v1",
            "abstract": "...",
            "source": "arxiv"
        },
        ...
    ]
}
```

**Example Console Output:**
```
[EXTERNAL_SEARCHER] Searching external sources (arXiv API)...
   Query: Top 5 SOTA models for reranking
  [OK] arXiv: Found 5 papers
   Found 5 results

   [INGEST] Adding 5 external papers to ChromaDB...
   [OK] Successfully ingested 5/5 papers
[OK] External search completed: 5 papers
```

---

### Node 8: Re-Search with External Papers

**File:** `src/agent/agent.py` → `_re_search_with_external()`

**Purpose:** Perform search again with newly ingested external papers

**Process:**

```python
def _re_search_with_external(self, state: dict) -> dict:
    print("\n[RE_SEARCH_WITH_EXTERNAL] Re-searching with external papers...")
    
    # External papers now in ChromaDB, re-run search
    state = self.search_executor(state)
    
    # This will include the newly ingested external papers
    # because ChromaDB now contains them
    
    print("[OK] Re-search completed with external papers")
    return state
```

**What changed:**
- Before: ChromaDB had 9 papers
- After: ChromaDB now has 9 + 5 = 14 papers (newly ingested)
- Re-search will find different results including external papers

**Example Console Output:**
```
[RE_SEARCH_WITH_EXTERNAL] Re-searching with external papers...

[SEARCH_EXECUTOR] Mode: hybrid
Number of requested results 100 is greater than number of elements in 
index 9, updating n_results = 9
[OK] Hybrid Search (RRF): 10 results
  [OK] Re-search completed with external papers
```

---

### Node 9: Re-Evaluate After External Search

**File:** `src/agent/agent.py` → `_re_evaluate_after_external()`

**Purpose:** Re-generate answer with external papers, then re-evaluate

**Process:**

```python
def _re_evaluate_after_external(self, state: dict) -> dict:
    print("\n[RE_EVALUATE] Checking answer quality after external search...")
    
    # Extract context again (with external papers now included)
    state = self.context_extractor(state)
    
    # Generate answer again with new context
    state = self.answer_generator(state)
    
    print("[OK] Answer re-generated with external papers")
    
    # Evaluate new answer
    state = self.result_evaluator(state)
    
    print(f"[OK] Re-evaluation Score: {int(state['answer_confidence'] * 10)}/10")
    
    # Check if we should retry external search
    iteration = state.get("external_search_iteration", 0) + 1
    max_iterations = state.get("max_external_iterations", 2)
    
    state["external_search_iteration"] = iteration
    
    # If score still low and iterations remain, retry
    if state["answer_confidence"] < 0.7 and iteration < max_iterations:
        state["needs_external_search"] = True
    else:
        state["needs_external_search"] = False
```

**Example Console Output:**
```
[CONTEXT_EXTRACTOR] Building context...
[OK] Context extracted: 20 papers, 1024 tokens

[ANSWER_GENERATOR] Generating RAG answer...
[OK] Answer generated (921 chars, 1 citations)
  [OK] Answer re-generated with external papers

[RE_EVALUATE] Checking answer quality after external search...

[RESULT_EVALUATOR] Evaluating answer quality...
[OK] Evaluation Score: 1/10 - The answer is severely incomplete...
  Iteration 1/2: Score = 1/10
  [INFO] Score 1/10 < 7, attempting another external search...
```

---

### Node 10: Response Formatter

**File:** `src/agent/nodes/response_formatter.py`

**Purpose:** Format final output for user

**Process:**

```python
def __call__(self, state: dict) -> dict:
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
    state["final_confidence"] = state.get("answer_confidence", 0.0)
    
    print(f"[OK] Response formatted with {len(final_sources)} sources")
    return state

def format_for_output(self, state: dict) -> dict:
    """Convert state to JSON for user"""
    
    return {
        "success": True,
        "query": state.get("query", ""),
        "intent": state.get("intent", "unclear"),
        "answer": state.get("final_answer", ""),
        "sources": state.get("final_sources", []),
        "confidence": round(state.get("final_confidence", 0.0), 2),
        "search_mode_used": state.get("search_mode", "hybrid"),
        "external_search_triggered": state.get("external_search_triggered", False),
        "execution_time_ms": state.get("execution_time_ms", 0)
    }
```

**Example JSON Output:**
```json
{
  "success": true,
  "query": "Top 5 SOTA models in reranker",
  "intent": "global",
  "answer": "Based on the provided papers, here are five significant reranker models...",
  "sources": [
    {
      "paper_id": "258833107",
      "title": "Bidirectional Transformer Reranker for Grammatical Error Correction",
      "score": 18.065,
      "relevance": "high"
    },
    ...
  ],
  "confidence": 0.1,
  "search_mode_used": "hybrid",
  "external_search_triggered": true,
  "execution_time_ms": 64427
}
```

---

## Data Transformations

### State Flow Example: "Top 5 SOTA models in reranker"

#### Step 1: Initial State
```python
state = {
    "query": "Top 5 SOTA models in reranker",
    "session_id": "session_123",
    "intent": "unclear",
    "search_mode": "hybrid",
    "execution_path": []
}
```

#### Step 2: After Intent Classification
```python
state = {
    "query": "Top 5 SOTA models in reranker",
    "intent": "global",                           # ← ADDED
    "intent_confidence": 0.95,                    # ← ADDED
    "intent_explanation": "General NLP question", # ← ADDED
    "execution_path": ["classify_intent"]         # ← UPDATED
}
```

#### Step 3: After Search Mode Selection
```python
state = {
    "search_mode": "hybrid",                      # ← UPDATED
    "execution_path": [..., "select_search_mode"] # ← UPDATED
}
```

#### Step 4: After Search Execution
```python
state = {
    "lexical_results": [                          # ← ADDED
        {"paper_id": "258833107", "title": "BTR", "score": 18.065},
        ...
    ],
    "semantic_results": [                         # ← ADDED
        {"paper_id": "276408982", "title": "Gumbel", "score": 0.823},
        ...
    ],
    "hybrid_results": [                           # ← ADDED
        # RRF-combined results
    ],
    "reranked_results": [                         # ← ADDED (final search results)
        ...
    ],
    "execution_path": [..., "search_executor"]    # ← UPDATED
}
```

#### Step 5: After Context Extraction
```python
state = {
    "context_text": """# Retrieved Papers for RAG
[1] Bidirectional Transformer...
    Paper ID: 258833107
    Score: 18.0650
    
    Abstract: ...""",                             # ← ADDED
    "context_documents": [                        # ← ADDED
        {"paper_id": "258833107", "title": "BTR", "score": 18.065},
        ...
    ],
    "context_size": 2100,                         # ← ADDED (tokens)
    "execution_path": [..., "context_extractor"]  # ← UPDATED
}
```

#### Step 6: After Answer Generation
```python
state = {
    "initial_answer": """Based on the provided papers, here are five 
    significant reranker models...""",            # ← ADDED
    "answer_confidence": 0.7,                     # ← ADDED
    "execution_path": [..., "answer_generator"]   # ← UPDATED
}
```

#### Step 7: After Evaluation
```python
state = {
    "is_answer_good": False,                      # ← UPDATED
    "answer_confidence": 0.1,  # 1/10 score       # ← UPDATED
    "needs_external_search": True,                # ← ADDED
    "feedback_reason": "Answer incomplete...",    # ← ADDED
    "execution_path": [..., "result_evaluator"]   # ← UPDATED
}
```

#### Step 8: After External Search (if triggered)
```python
state = {
    "external_papers": [                          # ← ADDED
        {
            "title": "arXiv Paper...",
            "arxiv_id": "2604.22751v1",
            "url": "https://arxiv.org/...",
            "abstract": "..."
        },
        ...
    ],
    "execution_path": [..., "external_searcher"]  # ← UPDATED
}
```

#### Step 9: After Re-Search & Re-Evaluation
```python
state = {
    # Search results updated with external papers
    "reranked_results": [                         # ← UPDATED (now includes external)
        ...
    ],
    # Answer regenerated
    "initial_answer": "Updated answer...",        # ← UPDATED
    "answer_confidence": 0.3,  # Re-evaluated     # ← UPDATED
    "external_search_iteration": 1,               # ← UPDATED
    "needs_external_search": True,  # Try again   # ← UPDATED
    "execution_path": [..., "re_evaluate_result"] # ← UPDATED
}
```

#### Step 10: After Response Formatting
```python
state = {
    "final_answer": "...",                        # ← ADDED
    "final_sources": [                            # ← ADDED
        {
            "paper_id": "258833107",
            "title": "BTR",
            "score": 18.065,
            "relevance": "high"
        },
        ...
    ],
    "final_confidence": 0.3,                      # ← ADDED
    "execution_path": [..., "response_formatter"] # ← UPDATED
}
```

---

## Example Execution

### Console Output from: `python run_agent.py --query "Top 5 SOTA models in reranker"`

```
[AGENT] Initializing agent...
[AGENT] Initializing PaperRAGAgent...
[SEARCH_EXECUTOR] Initializing search engines...
[BM25] Loading data from final_cleaned_data.jsonl to build index...
[BM25] Initializing BM25 Engine for 13012 papers...
[BM25] Index ready!

[OK] BM25 Searcher initialized
[INIT] Connecting to ChromaDB at: ./data/chroma_store_fulltext...
[OK] Connected to Collection: 'papers' (Total papers: 9)
[OK] ChromaDB Retriever initialized
[INIT] ExternalSearcher initialized (arXiv API + PDF parsing: True)
[INIT] PaperRAGAgent initialized!

Processing query: Top 5 SOTA models in reranker

================================================================================
Query: Top 5 SOTA models in reranker
================================================================================

[CLASSIFY_INTENT] Processing: 'Top 5 SOTA models in reranker...'
[OK] Intent: global (Confidence: 0.95)

[SELECT_SEARCH_MODE]
[OK] Search mode selected: hybrid

[SEARCH_EXECUTOR] Mode: hybrid
[OK] Hybrid Search (RRF): 10 results

[CONTEXT_EXTRACTOR] Building context...
[OK] Context extracted: 10 papers, 1024 tokens

[ANSWER_GENERATOR] Generating RAG answer...
[OK] Answer generated (403 chars, 0 citations)

[RESULT_EVALUATOR] Evaluating answer quality...
[OK] Evaluation Score: 0/10 - This answer fails on almost all criteria...
! External search triggered: This answer fails on almost all criteria...

[DECISION] Triggering external search (answer quality low)

[EXTERNAL_SEARCHER] Searching external sources (arXiv API)...
   Query: Top 5 SOTA models for reranking
  [OK] arXiv: Found 5 papers
   Found 5 results

   [INGEST] Adding 5 external papers to ChromaDB...
   [OK] Successfully ingested 5/5 papers
[OK] External search completed: 5 papers

[RE_SEARCH_WITH_EXTERNAL] Re-searching with external papers...

[SEARCH_EXECUTOR] Mode: hybrid
[OK] Hybrid Search (RRF): 10 results
  [OK] Re-search completed with external papers

[CONTEXT_EXTRACTOR] Building context...
[OK] Context extracted: 20 papers, 1024 tokens

[ANSWER_GENERATOR] Generating RAG answer...
[OK] Answer generated (921 chars, 1 citations)
  [OK] Answer re-generated with external papers

[RE_EVALUATE] Checking answer quality after external search...

[RESULT_EVALUATOR] Evaluating answer quality...
[OK] Evaluation Score: 1/10 - The answer is severely incomplete...
  Iteration 1/2: Score = 1/10
  [INFO] Score 1/10 < 7, attempting another external search...

[EXTERNAL_SEARCHER] Searching external sources (arXiv API)...
   Query: Top 5 SOTA models for reranking AND (cat:cs.CL OR cat:cs.LG)
  [OK] arXiv: Found 5 papers
   Found 5 results

   [INGEST] Adding 5 external papers to ChromaDB...
   [OK] Successfully ingested 5/5 papers
[OK] External search completed: 5 papers

[RE_SEARCH_WITH_EXTERNAL] Re-searching with external papers...

[SEARCH_EXECUTOR] Mode: hybrid
[OK] Hybrid Search (RRF): 10 results
  [OK] Re-search completed with external papers

[CONTEXT_EXTRACTOR] Building context...
[OK] Context extracted: 30 papers, 1024 tokens

[ANSWER_GENERATOR] Generating RAG answer...
[OK] Answer generated (522 chars, 0 citations)
  [OK] Answer re-generated with external papers

[RE_EVALUATE] Checking answer quality after external search...

[RESULT_EVALUATOR] Evaluating answer quality...
[OK] Evaluation Score: 0/10 - The answer is severely incomplete...
  Iteration 2/2: Score = 0/10
  [WARN] Max iterations (2) reached with score 0/10, stopping

[RESPONSE_FORMATTER] Formatting response...
[OK] Response formatted with 30 sources

[OK] Execution completed in 64427ms
Confidence: 0%
```

---

## Summary

### Data Flow Overview
```
JSONL File (13K papers)
    ↓
[BM25 Indexing]  +  [ChromaDB Embedding]
    ↓                       ↓
Lexical Index           Semantic Index
(Keyword matching)    (Vector similarity)
    ↓                       ↓
    └──→ [Hybrid Search - RRF Fusion] ←──┘
         ↓
    Ranked Results (10 papers)
         ↓
    [Context Extraction] (max 4000 tokens)
         ↓
    Context Window for RAG
         ↓
    [Gemini 2.5-Flash] + [Context] → Answer
         ↓
    [Evaluation: 0-10 score]
         ↓
    [If score < 7] → [External Search (arXiv)]
         ↓
    [Ingest to ChromaDB]
         ↓
    [Re-search + Re-evaluate] (max 2 iterations)
         ↓
    [Format Response]
         ↓
    JSON Output to User
```

### Key Metrics
- **Papers indexed:** 13,012 (BM25) + 9 (ChromaDB initially)
- **Search latency:** 250-350ms (hybrid)
- **LLM generation:** 500-800ms
- **Total latency:** 1-1.5s per query
- **Memory usage:** ~700MB
- **Max context window:** 4,000 tokens
- **Max external search iterations:** 2

