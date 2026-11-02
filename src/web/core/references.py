"""Normalize saved and in-memory chat references without changing user messages."""
import re


def normalize_references(content, sources):
    text = str(content or "")
    if not isinstance(sources, list) or not sources:
        return text, []
    labels = {}
    chunks = {}
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            continue
        try:
            label = int(source.get("ref_number") or index)
        except (TypeError, ValueError):
            label = index
        labels[label] = source
        if source.get("chunk_id"):
            chunks[str(source["chunk_id"])] = label
    text = re.sub(r"\[Chunk:\s*([^\]]+)\]",
                  lambda m: f"[{chunks[m[1].strip()]}]" if m[1].strip() in chunks else m[0],
                  text, flags=re.IGNORECASE)
    order = list(dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", text)
                               if int(n) in labels))
    order.extend(n for n in labels if n not in order)
    papers, mapping, normalized = {}, {}, []
    for label in order:
        source = labels[label]
        paper_id = str(source.get("paper_id") or "").strip()
        title = " ".join(str(source.get("title") or "").split()).casefold()
        url = str(source.get("source_url") or "").strip()
        if paper_id and paper_id.casefold() not in {"n/a", "unknown", "none"}:
            key = ("paper", paper_id)
        elif title and title not in {"untitled", "unknown title", "n/a"}:
            key = ("title", title)
        elif url:
            key = ("url", url)
        else:
            key = ("chunk", source.get("chunk_id") or label)
        if key not in papers:
            papers[key] = len(normalized) + 1
            normalized.append(dict(source, ref_number=papers[key]))
        mapping[label] = papers[key]
    text = re.sub(r"\[(\d+)\]", lambda m: f"[{mapping[int(m[1])]}]"
                  if int(m[1]) in mapping else m[0], text)
    text = re.sub(r"(?:\[\d+\])+",
                  lambda m: "".join(dict.fromkeys(re.findall(r"\[\d+\]", m[0]))), text)
    return text, normalized
