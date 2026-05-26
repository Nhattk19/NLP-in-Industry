# -*- coding: utf-8 -*-
"""
External Search Node
Search for new papers from arXiv API when database is outdated or answer is insufficient
"""

import os
import json
import re
import time
import requests
from typing import List, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr
from dotenv import load_dotenv
from urllib.parse import quote
import xml.etree.ElementTree as ET
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent.states import IntentType
from src.chroma_fulltext.ingest import (
    BATCH_SIZE,
    EMBED_BATCH_SIZE,
    MIN_FULLTEXT_WORDS,
    build_metadata,
    load_embedding_model,
    split_document,
    COLLECTION_NAME as FULLTEXT_COLLECTION_NAME,
)

# Try to import pdfplumber for PDF parsing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", GEMINI_MODEL)
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
EXTERNAL_SEARCH_NUM_RESULTS = int(os.getenv("EXTERNAL_SEARCH_NUM_RESULTS", 5))
EXTERNAL_SEARCH_CANDIDATE_RESULTS = 10
EXTERNAL_SEARCH_MAX_PAGES = 1
PDF_PARSER_ENABLED = os.getenv("PDF_PARSER_ENABLED", "true").lower() == "true"
REQUEST_TIMEOUT = 15


class ExternalSearcher:
    """Search external sources (arXiv API) and parse PDFs"""
    
    def __init__(self):
        """Initialize external searcher"""
        self.max_results = min(max(EXTERNAL_SEARCH_NUM_RESULTS, 1), 5)
        self.candidate_results = EXTERNAL_SEARCH_CANDIDATE_RESULTS
        self.max_pages = EXTERNAL_SEARCH_MAX_PAGES
        self.enable_pdf_parsing = PDF_PARSER_ENABLED
        self.arxiv_base_url = "http://export.arxiv.org/api/query?"
        self.query_rewriter = None

        if ENABLE_QUERY_REWRITE and GOOGLE_API_KEY:
            try:
                self.query_rewriter = ChatGoogleGenerativeAI(
                    model=QUERY_REWRITE_MODEL,
                    api_key=GOOGLE_API_KEY,
                    temperature=0,
                    max_output_tokens=256,
                )
            except Exception as e:
                print(f"! [INIT] Query rewriter unavailable: {e}")

        print(
            f"[INIT] ExternalSearcher initialized "
            f"(target={self.max_results}, candidates={self.candidate_results}, "
            f"pages={self.max_pages}, PDF parsing: {self.enable_pdf_parsing}, "
            f"query rewrite: {bool(self.query_rewriter)})"
        )
    
    def __call__(self, state: dict) -> dict:
        """Execute external search with arXiv API"""
        
        print("\n[EXTERNAL_SEARCHER] Searching external sources (arXiv API)...")
        
        # Build search query
        search_query = self._build_search_query(state)
        if state.get("rewritten_search_query"):
            print(f"   Rewritten: {state['rewritten_search_query']}")
        print(f"   Query: {search_query}")
        
        try:
            # Search only the first 10 candidates. We do not fetch beyond this
            # window even if fewer than 5 papers survive DB filtering.
            results = self._collect_candidate_papers(search_query)
            
            print(f"   Found {len(results)} results")

            # Avoid downloading/parsing PDFs for papers already in ChromaDB.
            # arXiv gives stable paper_id values in the metadata response, so
            # we can cheaply dedupe before the expensive crawl step.
            results, skipped_existing = self._remove_existing_db_papers(
                results,
                max_selected=self.candidate_results,
            )
            if skipped_existing:
                print(f"   [DB] Skipped {len(skipped_existing)} papers already present in DB before PDF parsing")
            print(f"   [DB] {len(results)} new candidate papers remain before PDF parsing")

            # Tighten the result set before PDF parsing so unrelated arXiv
            # papers do not waste crawl/parse time.
            results = self._filter_and_rank_external_results(results, state.get("query", ""))
            
            # Parse PDFs if enabled
            if self.enable_pdf_parsing and results:
                results = self._enrich_with_pdf_content(results)

            # Re-rank after PDF enrichment, then run a final duplicate check
            # just before ingestion in case another job inserted the paper.
            results = self._filter_and_rank_external_results(results, state.get("query", ""))
            results, skipped_after_parse = self._remove_existing_db_papers(results)
            if skipped_after_parse:
                print(f"   [DB] Skipped {len(skipped_after_parse)} papers already present in DB after PDF parsing")
            print(f"   [DB] {len(results)} new papers remain after DB filtering")
            
            # Ingest results into ChromaDB for future searches
            if results:
                state["external_papers"] = self._ingest_to_chromadb(results, state.get("query", ""))
                state["bm25_index_updated"] = bool(state["external_papers"])
            else:
                state["external_papers"] = []
                state["bm25_index_updated"] = False
            
            print(f"[OK] External search completed: {len(state['external_papers'])} chunks")
            
        except Exception as e:
            print(f"X External search failed: {str(e)}")
            state["external_papers"] = []
        
        state["execution_path"] = state.get("execution_path", []) + ["external_searcher"]
        return state

    def _get_chromadb_collection(self):
        """Open the shared ChromaDB collection used by the agent."""
        import chromadb
        from chromadb.config import Settings

        os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        CHROMA_PATH = "./data/chroma_store_fulltext"
        settings = Settings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=settings)
        return client.get_or_create_collection(
            name=FULLTEXT_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _collect_candidate_papers(self, search_query: str) -> List[Dict]:
        """Fetch the first 10 arXiv candidates before DB filtering."""
        collected = []
        seen_ids = set()

        page_results = self._search_arxiv(
            search_query,
            start=0,
            max_results=self.candidate_results,
        )

        if not page_results:
            return collected

        for paper in page_results:
            paper_id = paper.get("paper_id") or paper.get("url") or paper.get("title") or ""
            if not paper_id or paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            collected.append(paper)

            if len(collected) >= self.candidate_results:
                break

        return collected
    
    def _build_search_query(self, state: dict) -> str:
        """Build search query for arXiv - improved with iteration-aware refinement"""
        
        # Get base query
        refined_query = state.get("refined_query", "")
        base_query = refined_query or state.get("query", "")
        base_query = base_query.strip()

        rewritten_query = self._rewrite_search_query(base_query)
        if rewritten_query:
            state["rewritten_search_query"] = rewritten_query
            search_query = rewritten_query

            iteration = state.get("external_search_iteration", 0)
            if iteration == 1 and "cat:" not in self._normalize_query_text(search_query):
                search_query = f"{search_query} AND (cat:cs.CL OR cat:cs.LG)"
            elif iteration >= 2 and "cat:" not in self._normalize_query_text(search_query):
                search_query = f"{search_query} AND (cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:cs.NE)"

            return search_query[:200]

        normalized_base_query = self._normalize_query_text(base_query)
        if self._is_sota_ner_query(normalized_base_query):
            # Special-case SOTA-for-NER queries: search directly for the task
            # name and transformer-style NER terms, instead of generic SOTA
            # wording that matches many unrelated recent papers.
            return (
                '(ti:"named entity recognition" OR abs:"named entity recognition" OR '
                'all:"named entity recognition" OR all:NER) AND '
                '(cat:cs.CL OR cat:cs.LG OR cat:cs.AI) AND '
                '(all:transformer OR all:BERT OR all:RoBERTa OR all:ELECTRA OR '
                'all:"sequence tagging" OR all:"token classification")'
            )

        query_parts = [base_query]
        
        # Iteration-based query refinement
        iteration = state.get("external_search_iteration", 0)
        if iteration == 1:
            # First attempt: focused on NLP/ML
            query_parts.append("AND (cat:cs.CL OR cat:cs.LG)")
        elif iteration >= 2:
            # Second attempt: broaden to include AI/NE
            query_parts.append("AND (cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:cs.NE)")

        search_query = " ".join(query_parts)
        return search_query[:200]  # Limit length

    def _rewrite_search_query(self, query: str) -> str:
        """Rewrite a natural-language question into a compact academic search query."""
        if not self.query_rewriter:
            return ""

        query = (query or "").strip()
        if not query:
            return ""

        prompt = f"""Convert the user question into a concise arXiv search query for academic papers.

Return valid JSON only, with this schema:
{{
  "search_query": "string",
  "reason": "string"
}}

Rules:
- Keep the query short and targeted.
- Prefer academic terms, paper titles, techniques, and task names.
- If the question is broad, rewrite it to the most likely research topic.
- If the question is about Transformer, include terms like "Transformer architecture", self-attention, encoder-decoder.
- If the question is about SOTA for NER, include named entity recognition / NER and strong model families such as BERT, RoBERTa, or DeBERTa.
- Add category filters only when they help, such as cat:cs.CL, cat:cs.LG, or cat:cs.AI.
- Do not answer the question. Only output the search query.

User question:
{query}"""

        try:
            response = self.query_rewriter.invoke(prompt)
            raw_text = getattr(response, "content", "") or ""
            raw_text = raw_text.strip()

            if "```json" in raw_text:
                raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```", 1)[1].split("```", 1)[0].strip()

            payload = json.loads(raw_text)
            search_query = str(payload.get("search_query", "")).strip()
            return search_query[:200]
        except Exception as e:
            print(f"   [REWRITE] Query rewrite failed: {e}")
            return ""

    def _normalize_query_text(self, text: str) -> str:
        """Lowercase and collapse whitespace for lightweight query checks."""
        return re.sub(r"\s+", " ", text or "").strip().lower()

    def _is_sota_ner_query(self, query: str) -> bool:
        """Detect SOTA queries that are actually asking about NER."""
        return (
            ("sota" in query or "state of the art" in query)
            and ("ner" in query or "named entity recognition" in query)
        )

    def _score_external_result(self, paper: Dict, query: str) -> float:
        """Score how well an external paper matches the query."""
        text = self._normalize_query_text(
            " ".join(
                [
                    str(paper.get("title", "")),
                    str(paper.get("snippet", "")),
                    str(paper.get("pdf_content", "")),
                ]
            )
        )

        score = 0.0
        if self._is_sota_ner_query(query):
            if "named entity recognition" in text:
                score += 5.0
            if "ner" in f" {text} ":
                score += 2.5
            for term in ("transformer", "bert", "roberta", "electra", "token classification", "sequence tagging"):
                if term in text:
                    score += 0.75
            if any(term in text for term in ("named entity", "entity recognition", "sequence labeling")):
                score += 1.5
        else:
            query_tokens = [tok for tok in re.findall(r"\w+", query) if len(tok) > 2]
            for token in query_tokens:
                if token in text:
                    score += 0.5

        return score

    def _filter_and_rank_external_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Keep only the most relevant external results before ingestion."""
        if not results:
            return results

        normalized_query = self._normalize_query_text(query)
        scored_results = []

        for item in results:
            score = self._score_external_result(item, normalized_query)
            item = item.copy()
            item["external_match_score"] = round(score, 4)
            scored_results.append((score, item))

        # For SOTA/NER queries, drop obviously unrelated papers.
        if self._is_sota_ner_query(normalized_query):
            scored_results = [pair for pair in scored_results if pair[0] >= 2.5]

        scored_results.sort(key=lambda pair: (pair[0], pair[1].get("rank", 0)), reverse=True)

        refined = []
        for idx, (score, item) in enumerate(scored_results[: self.max_results], start=1):
            item["rank"] = idx
            item["relevance_score"] = float(score if score > 0 else 1.0 / idx)
            refined.append(item)

        return refined or results[: self.max_results]
    
    def _search_arxiv(self, query: str, start: int = 0, max_results: int = None) -> List[Dict]:
        """Search using arXiv API"""
        
        results = []
        request_results = max_results or self.max_results
        
        try:
            normalized_query = self._normalize_query_text(query)
            targeted_search = self._is_sota_ner_query(normalized_query)
            sort_by = "relevance" if targeted_search else "submittedDate"
            
            # arXiv API query
            search_url = (
                self.arxiv_base_url
                + f"search_query={quote(query)}&start={start}&max_results={request_results}"
                + f"&sortBy={sort_by}&sortOrder=descending"
            )
            
            response = requests.get(search_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Register namespaces properly - try multiple approaches
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Try to find entries with atom namespace first
            entries = root.findall('atom:entry', ns)
            
            # If no entries found with standard namespace, try bare tags
            if not entries:
                entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            
            # If still no entries, try arxiv namespace
            if not entries:
                entries = root.findall('{http://arxiv.org/schemas/atom}entry')
            
            # Last resort: try without any namespace
            if not entries:
                entries = root.findall('entry')
            
            for idx, entry in enumerate(entries[:request_results]):
                try:
                    # Extract fields with proper namespace handling
                    title_elem = entry.find('atom:title', ns)
                    if title_elem is None:
                        title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Unknown"
                    
                    summary_elem = entry.find('atom:summary', ns)
                    if summary_elem is None:
                        summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                    summary = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
                    
                    # Get paper ID
                    id_elem = entry.find('atom:id', ns)
                    if id_elem is None:
                        id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                    arxiv_id = ""
                    if id_elem is not None and id_elem.text:
                        arxiv_id = id_elem.text.split('/abs/')[-1] if '/abs/' in id_elem.text else id_elem.text
                    
                    # Get PDF link
                    pdf_url = ""
                    links = entry.findall('atom:link', ns)
                    if not links:
                        links = entry.findall('{http://www.w3.org/2005/Atom}link')
                    for link in links:
                        if link.get('type') == 'application/pdf':
                            pdf_url = link.get('href', '')
                            break
                    
                    # Get authors
                    authors = []
                    author_elems = entry.findall('atom:author', ns)
                    if not author_elems:
                        author_elems = entry.findall('{http://www.w3.org/2005/Atom}author')
                    for author in author_elems[:3]:
                        name_elem = author.find('atom:name', ns)
                        if name_elem is None:
                            name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                        if name_elem is not None and name_elem.text:
                            authors.append(name_elem.text)
                    
                    result = {
                        "title": title,
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "pdf_url": pdf_url,
                        "snippet": summary[:300],
                        "source": "arxiv",
                        "paper_id": f"arxiv:{arxiv_id}",
                        "rank": idx + 1,
                        "relevance_score": 1.0 / (idx + 1),
                        "is_paper": True,
                        "authors": authors,
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"      [WARN] Error parsing entry: {str(e)[:50]}")
                    continue
            
            if results:
                print(f"  [OK] arXiv: Found {len(results)} papers")
            else:
                print(f"  [WARN] arXiv: No results found")
            
        except requests.RequestException as e:
            print(f"  [ERR] arXiv API error: {str(e)[:80]}")
        except ET.ParseError as e:
            print(f"  [ERR] XML parse error: {str(e)[:80]}")
        except Exception as e:
            print(f"  [ERR] Unexpected error: {str(e)[:80]}")
        
        return results

    def _paper_exists_in_db(self, collection, paper_id: str) -> bool:
        """Check whether a paper already exists in the shared ChromaDB collection."""
        if not paper_id:
            return False

        try:
            existing = collection.get(
                where={"paper_id": paper_id},
                limit=1,
                include=["metadatas"],
            )
            return bool(existing.get("ids"))
        except Exception as e:
            print(f"      [WARN] DB existence check failed for {paper_id}: {str(e)}")
            return False

    def _remove_existing_db_papers(self, results: List[Dict], max_selected: Optional[int] = None):
        """Keep new papers that are not already stored in ChromaDB."""
        if not results:
            return [], []

        max_selected = max_selected or self.max_results

        try:
            collection = self._get_chromadb_collection()
        except Exception as e:
            print(f"   [WARN] Could not open ChromaDB for duplicate check: {str(e)}")
            return results[:max_selected], []

        selected = []
        skipped = []
        seen_paper_ids = set()

        for paper in results:
            paper_id = paper.get("paper_id", "") or ""
            if not paper_id or paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper_id)

            if self._paper_exists_in_db(collection, paper_id):
                skipped.append(paper)
                continue

            selected.append(paper)
            if len(selected) >= max_selected:
                break

        return selected, skipped

    def _is_paper_url(self, url: str) -> bool:
        """Check if URL is a paper link"""
        
        paper_domains = [
            "arxiv.org",
            "scholar.google.com",
            "researchgate.net",
            "semanticscholar.org",
            "acl-arc.org",
            "aclanthology.org",
            "proceedings.neurips.cc",
            "icml.cc",
            "iclr.cc",
            "openreview.net"
        ]
        
        url_lower = url.lower()
        return any(domain in url_lower for domain in paper_domains)
    
    def _extract_paper_id(self, url: str) -> Optional[str]:
        """Extract paper ID from URL"""
        
        # arXiv format: arxiv.org/abs/XXXX.XXXXX
        arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', url)
        if arxiv_match:
            return f"arxiv:{arxiv_match.group(1)}"
        
        # DOI format
        doi_match = re.search(r'doi\.org/(10\.\d+/[^\s]+)', url)
        if doi_match:
            return f"doi:{doi_match.group(1)}"
        
        return None
    
    def _enrich_with_pdf_content(self, results: List[Dict]) -> List[Dict]:
        """Download and parse PDFs to extract content"""
        
        if not PDFPLUMBER_AVAILABLE:
            return results
        
        enriched = []
        
        for item in results:
            if not item.get("is_paper"):
                enriched.append(item)
                continue
            
            # Try to get PDF URL
            pdf_url = self._get_pdf_url(item["url"])
            
            if pdf_url:
                print(f"   [PDF] Parsing: {item['title'][:50]}...")
                
                try:
                    # Download PDF
                    pdf_content = self._download_pdf(pdf_url)
                    
                    if pdf_content:
                        # Parse PDF
                        text = self._parse_pdf(pdf_content)
                        
                        if text:
                            item["pdf_content"] = text[:2000]  # First 2000 chars
                            item["content_source"] = "pdf"
                            print(f"      [OK] Extracted {len(text)} characters")
                
                except Exception as e:
                    print(f"      [WARN] PDF parsing failed: {str(e)}")
            
            enriched.append(item)
        
        return enriched
    
    def _get_pdf_url(self, page_url: str) -> Optional[str]:
        """Get PDF URL from paper page"""
        
        # Direct PDF link
        if page_url.endswith(".pdf"):
            return page_url
        
        # arXiv: convert /abs/ to /pdf/
        if "arxiv.org/abs/" in page_url:
            return page_url.replace("/abs/", "/pdf/") + ".pdf"
        
        # ResearchGate: try to extract PDF
        if "researchgate.net" in page_url:
            # ResearchGate usually has download links in HTML
            return None
        
        return None
    
    def _download_pdf(self, url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[bytes]:
        """Download PDF from URL"""
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not url.endswith(".pdf"):
                return None
            
            # Limit size to 50MB
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > 50 * 1024 * 1024:
                return None
            
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 50 * 1024 * 1024:  # 50MB limit
                    return None
            
            return content
        
        except requests.RequestException as e:
            print(f"      Download error: {str(e)}")
            return None
    
    def _parse_pdf(self, pdf_content: bytes) -> Optional[str]:
        """Parse PDF and extract text"""
        
        if not PDFPLUMBER_AVAILABLE:
            return None
        
        try:
            from io import BytesIO
            
            pdf_file = BytesIO(pdf_content)
            
            with pdfplumber.open(pdf_file) as pdf:
                text_parts = []
                
                # Extract from first 5 pages
                for page_num, page in enumerate(pdf.pages[:5]):
                    try:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                    except Exception as e:
                        print(f"      Page {page_num} error: {str(e)}")
                
                full_text = "\n".join(text_parts)
                
                # Clean up text
                full_text = re.sub(r'\s+', ' ', full_text)
                full_text = full_text.strip()
                
                return full_text if full_text else None
        
        except Exception as e:
            print(f"      PDF parsing error: {str(e)}")
            return None

    def _prepare_external_chunk_records(self, results: List[Dict], query: str):
        """Convert external paper results into chunk-level records."""
        chunk_records = []
        documents = []
        metadatas = []
        ids = []
        successful_papers = 0

        for idx, paper in enumerate(results):
            try:
                paper_id = paper.get("paper_id") or f"external_{query}_{idx}"
                title = paper.get("title", "")
                snippet = paper.get("snippet", "")
                pdf_content = paper.get("pdf_content", "")
                source_url = paper.get("url", "")
                source = paper.get("source", "external")
                rank = int(paper.get("rank", idx + 1))
                score = float(paper.get("relevance_score", 1.0 / (idx + 1)))
                pdf_url = paper.get("pdf_url", "")

                full_text_parts = [
                    part.strip()
                    for part in [title, snippet, pdf_content]
                    if part and part.strip()
                ]
                full_text = "\n\n".join(full_text_parts).strip()

                if not full_text:
                    continue

                if len(full_text.split()) < MIN_FULLTEXT_WORDS:
                    continue

                chunks = split_document(full_text)
                if not chunks:
                    continue

                successful_papers += 1

                for chunk_index, chunk in enumerate(chunks):
                    chunk_id = f"{paper_id}_chunk_{chunk_index:04d}"
                    chunk_text = chunk.text or ""
                    metadata_record = {
                        "paper_id": paper_id,
                        "title": title,
                        "source_url": source_url,
                    }

                    documents.append(chunk_text)
                    metadatas.append(build_metadata(metadata_record, chunk_index, chunk))
                    ids.append(chunk_id)

                    chunk_records.append({
                        "paper_id": paper_id,
                        "chunk_id": chunk_id,
                        "title": title,
                        "source_url": source_url,
                        "source": source,
                        "pdf_url": pdf_url,
                        "rank": rank,
                        "score": score,
                        "similarity": score,
                        "chunk_index": int(chunk_index),
                        "chunk_start": int(chunk.start),
                        "chunk_length": int(len(chunk_text)),
                        "chunk_text": chunk_text,
                        "text": chunk_text,
                        "snippet": chunk_text[:200],
                        "paper_snippet": snippet[:300],
                    })

            except Exception as e:
                print(f"      [WARN] Failed to prepare paper {idx}: {str(e)}")

        return chunk_records, documents, metadatas, ids, successful_papers

    def _upsert_chunk_batch(self, collection, model, documents, metadatas, ids):
        """Encode a batch of chunks and upsert them into ChromaDB."""
        embeddings = model.encode(
            documents,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
        ).tolist()

        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
    
    def _ingest_to_chromadb(self, results: List[Dict], query: str):
        """Ingest external papers into ChromaDB and return chunk-level records"""
        
        chunk_records, documents, metadatas, ids, successful_papers = self._prepare_external_chunk_records(results, query)
        if not chunk_records:
            print("   [WARN] No chunk-level content prepared for ingestion")
            return []

        try:
            # Connect to ChromaDB
            collection = self._get_chromadb_collection()
            model = load_embedding_model()

            print(
                f"\n   [INGEST] Adding {successful_papers} external papers to ChromaDB as "
                f"{len(chunk_records)} external chunks..."
            )

            inserted_chunks = 0
            for batch_start in range(0, len(documents), BATCH_SIZE):
                batch_end = batch_start + BATCH_SIZE
                batch_documents = documents[batch_start:batch_end]
                batch_metadatas = metadatas[batch_start:batch_end]
                batch_ids = ids[batch_start:batch_end]
                self._upsert_chunk_batch(collection, model, batch_documents, batch_metadatas, batch_ids)
                inserted_chunks += len(batch_ids)

            print(
                f"   [OK] Successfully ingested {successful_papers}/{len(results)} papers "
                f"as {inserted_chunks} chunks"
            )
            return chunk_records

        except ImportError:
            print("   [WARN] ChromaDB not available for ingestion")
            return chunk_records
        except Exception as e:
            print(f"   ⚠️  ChromaDB ingestion error: {str(e)}")
            return chunk_records
    
def get_external_searcher() -> ExternalSearcher:
    """Singleton getter for external searcher"""
    if not hasattr(get_external_searcher, '_instance'):
        get_external_searcher._instance = ExternalSearcher()
    return get_external_searcher._instance
