"""Resolve paper links from search results and the full-text citation store."""


def matches_detail_key(paper: dict, key: str) -> bool:
    return any(str(paper.get(field) or "").strip() == key
               for field in ("paper_id", "doi", "arxiv", "title"))


def resolve_detail_paper(key: str, cached_results: list, metadata: list, fulltext_collection=None):
    for paper in [*(cached_results or []), *(metadata or [])]:
        if matches_detail_key(paper, key):
            return paper
    # External and full-text-only papers are absent from the abstract index.
    if fulltext_collection is not None:
        for field in ("paper_id", "title"):
            result = fulltext_collection.get(where={field: key}, limit=1,
                                              include=["metadatas"])
            records = result.get("metadatas") or []
            if records:
                paper = dict(records[0] or {})
                paper.setdefault("abstract", "No abstract is available for this paper. Full-text Q&A is available below.")
                url = str(paper.get("source_url") or "")
                if "arxiv.org/" in url:
                    paper["arxiv"] = url
                elif "doi.org/" in url:
                    paper["doi"] = url
                elif "semanticscholar.org/" in url:
                    paper["s2_url"] = url
                return paper
    return None
