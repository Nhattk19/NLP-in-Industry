# -*- coding: utf-8 -*-
"""
External Search Node
Search for new papers from arXiv API when database is outdated or answer is insufficient
"""

import os
import re
import time
import requests
from typing import List, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr
from dotenv import load_dotenv
from urllib.parse import urljoin, quote
import xml.etree.ElementTree as ET

from src.agent.states import IntentType

# Try to import pdfplumber for PDF parsing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False

load_dotenv()

EXTERNAL_SEARCH_NUM_RESULTS = int(os.getenv("EXTERNAL_SEARCH_NUM_RESULTS", 5))
PDF_PARSER_ENABLED = os.getenv("PDF_PARSER_ENABLED", "true").lower() == "true"
REQUEST_TIMEOUT = 15


class ExternalSearcher:
    """Search external sources (arXiv API) and parse PDFs"""
    
    def __init__(self):
        """Initialize external searcher"""
        self.max_results = EXTERNAL_SEARCH_NUM_RESULTS
        self.enable_pdf_parsing = PDF_PARSER_ENABLED
        self.arxiv_base_url = "http://export.arxiv.org/api/query?"
        
        print(f"[INIT] ExternalSearcher initialized (arXiv API + PDF parsing: {self.enable_pdf_parsing})")
    
    def __call__(self, state: dict) -> dict:
        """Execute external search with arXiv API"""
        
        print("\n[EXTERNAL_SEARCHER] Searching external sources (arXiv API)...")
        
        # Build search query
        search_query = self._build_search_query(state)
        print(f"   Query: {search_query}")
        
        try:
            # Search arXiv (no rate limiting issues)
            results = self._search_arxiv(search_query)
            
            print(f"   Found {len(results)} results")
            
            # Parse PDFs if enabled
            if self.enable_pdf_parsing and results:
                results = self._enrich_with_pdf_content(results)
            
            # Ingest results into ChromaDB for future searches
            if results:
                self._ingest_to_chromadb(results, state.get("query", ""))
            
            state["external_papers"] = results
            
            print(f"[OK] External search completed: {len(results)} papers")
            
        except Exception as e:
            print(f"X External search failed: {str(e)}")
            state["external_papers"] = []
        
        state["execution_path"] = state.get("execution_path", []) + ["external_searcher"]
        return state
    
    def _build_search_query(self, state: dict) -> str:
        """Build search query for arXiv - improved with iteration-aware refinement"""
        
        query_parts = []
        
        # Get base query
        refined_query = state.get("refined_query", "")
        if refined_query:
            query_parts.append(refined_query)
        else:
            query_parts.append(state.get("query", ""))
        
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
    
    def _search_arxiv(self, query: str) -> List[Dict]:
        """Search using arXiv API"""
        
        results = []
        
        try:
            # arXiv API query
            search_url = self.arxiv_base_url + f"search_query={quote(query)}&start=0&max_results={self.max_results}&sortBy=submittedDate&sortOrder=descending"
            
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
            
            for idx, entry in enumerate(entries[:self.max_results]):
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
    
    def _ingest_to_chromadb(self, results: List[Dict], query: str):
        """Ingest external papers into ChromaDB"""
        
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            # Disable ChromaDB telemetry to avoid telemetry errors
            os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
            
            # Connect to ChromaDB
            CHROMA_PATH = "./data/chroma_store_fulltext"
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            emb_fn = embedding_functions.DefaultEmbeddingFunction()
            
            collection = client.get_or_create_collection(
                name="papers",
                metadata={"hnsw:space": "cosine"},
                embedding_function=emb_fn
            )
            
            print(f"\n   [INGEST] Adding {len(results)} external papers to ChromaDB...")
            
            successful = 0
            for idx, paper in enumerate(results):
                try:
                    # Generate unique ID for external papers
                    paper_id = paper.get("paper_id") or f"external_{query}_{idx}"
                    
                    # Build document text
                    title = paper.get("title", "")
                    snippet = paper.get("snippet", "")
                    pdf_content = paper.get("pdf_content", "")
                    
                    # Combine all text
                    doc_text = f"{title}. {snippet}"
                    if pdf_content:
                        doc_text += f"\n{pdf_content}"
                    
                    if not doc_text.strip():
                        continue
                    
                    # Build metadata
                    metadata = {
                        "title": title[:500],
                        "source": paper.get("source", "external"),
                        "url": paper.get("url", ""),
                        "rank": str(paper.get("rank", idx + 1)),
                        "is_external": "true",
                        "query_context": query[:100]
                    }
                    
                    # Add to collection (suppress ChromaDB logging)
                    with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
                        collection.add(
                            ids=[paper_id],
                            documents=[doc_text],
                            metadatas=[metadata]
                        )
                    
                    successful += 1
                    
                except Exception as e:
                    print(f"      [WARN] Failed to ingest paper {idx}: {str(e)}")
            
            print(f"   [OK] Successfully ingested {successful}/{len(results)} papers")
            
        except ImportError:
            print("   [WARN] ChromaDB not available for ingestion")
        except Exception as e:
            print(f"   ⚠️  ChromaDB ingestion error: {str(e)}")
    
def get_external_searcher() -> ExternalSearcher:
    """Singleton getter for external searcher"""
    if not hasattr(get_external_searcher, '_instance'):
        get_external_searcher._instance = ExternalSearcher()
    return get_external_searcher._instance
