"""Retrieval — search, deduplicate, novelty scoring."""

from mlx_lm.web_search import search_web, is_relevant
from research_agent.memory import ResearchMemory


def seed_search(topic: str, num_results: int = 15) -> list[dict]:
    """Broad initial search across multiple broad queries."""
    queries = [topic, f"{topic} overview", f"{topic} history"]
    seen_urls = set()
    candidates = []
    for q in queries:
        results = search_web(q, num_results=5)
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls and is_relevant(r.get("title", ""), r.get("snippet", ""), topic):
                seen_urls.add(url)
                r["source_query"] = q
                r["iteration"] = 0
                candidates.append(r)
    return candidates[:num_results]


def search_queries(queries: list[dict], memory: ResearchMemory) -> list[dict]:
    """Search for each query, skip URLs already seen."""
    new_candidates = []
    for q_item in queries:
        q = q_item["query"]
        if q in memory.searched_queries:
            continue
        memory.searched_queries.append(q)
        results = search_web(q, num_results=5)
        for r in results:
            url = r.get("url", "")
            if not url or url in memory.accepted_urls:
                continue
            if not is_relevant(r.get("title", ""), r.get("snippet", ""), memory.topic):
                continue
            r["source_query"] = q
            r["dimension"] = q_item.get("dimension", "overview")
            r["mode"] = q_item.get("mode", "exploit")
            r["iteration"] = memory.iteration
            r["novelty"] = novelty_score(r, memory)
            new_candidates.append(r)
    return new_candidates


def novelty_score(doc: dict, memory: ResearchMemory) -> float:
    """Score how much new information a doc provides.
    
    Based on new entities, concepts, dimensions not yet covered.
    0.0 = completely redundant, 1.0 = completely novel.
    """
    title = (doc.get("title", "") + " " + doc.get("snippet", "")).lower()
    words = set(title.split())

    # Entity overlap
    entity_overlap = sum(1 for e in memory.entities_seen if e.lower() in title)
    has_new_entities = entity_overlap == 0

    # Concept overlap
    concept_overlap = sum(1 for c in memory.concepts_seen if c.lower() in title)
    has_new_concepts = concept_overlap == 0

    # Dimension gap fill
    dim = doc.get("dimension", "")
    is_new_dimension = dim and memory.dimensions.get(dim, 1.0) < 0.3

    # Already seen URLs in candidate pool
    seen_before = any(
        d.get("url") == doc.get("url") or
        d.get("title", "").lower() in title
        for d in memory.candidate_docs
    )

    score = 0.0
    if has_new_entities:
        score += 0.3
    if has_new_concepts:
        score += 0.3
    if is_new_dimension:
        score += 0.25
    if not seen_before:
        score += 0.15

    return min(1.0, score)


def score_candidate(doc: dict, memory: ResearchMemory) -> dict:
    """Full candidate scoring: authority + novelty + gap fill + diversity."""
    url = doc.get("url", "").lower()
    authority = _authority_score(url, doc.get("title", ""))
    novelty = doc.get("novelty", novelty_score(doc, memory))
    dim = doc.get("dimension", "")
    gap_fill = 0.25 if dim and memory.dimensions.get(dim, 1.0) < 0.3 else 0.0
    diversity = _diversity_score(url, memory)

    doc["score"] = (
        0.30 * authority +
        0.30 * novelty +
        0.25 * gap_fill +
        0.10 * diversity +
        0.05 * 0.5  # recency (not calculated)
    )
    doc["authority"] = authority
    return doc


def _authority_score(url: str, title: str) -> float:
    """Heuristic authority score based on domain."""
    if "wikipedia.org" in url:
        return 1.0
    if "britannica.com" in url:
        return 1.0
    if url.endswith(".gov") or ".gov/" in url:
        return 0.9
    if url.endswith(".edu") or ".edu/" in url:
        return 0.85
    if any(d in url for d in ["reuters.com", "ap.org", "bbc.com", "bbc.co.uk",
                               "nytimes.com", "wsj.com", "economist.com"]):
        return 0.8
    if any(d in url for d in ["nature.com", "science.org", "sciencedirect.com",
                               "academic.oup.com", "springer.com", "jstor.org"]):
        return 0.9
    if url.endswith(".org") or ".org/" in url:
        return 0.5
    return 0.3


def _diversity_score(url: str, memory: ResearchMemory) -> float:
    """Penalize same-domain saturation."""
    domain = url.split("/")[2] if "://" in url else url.split("/")[0]
    same_domain = sum(1 for d in memory.candidate_docs
                      if domain in d.get("url", ""))
    if same_domain >= 3:
        return 0.1
    if same_domain >= 1:
        return 0.5
    return 1.0


def deduplicate_candidates(candidates: list[dict], memory: ResearchMemory) -> list[dict]:
    """Deduplicate by URL and basic semantic similarity."""
    seen_titles = set()
    deduped = []
    for doc in candidates:
        url = doc.get("url", "")
        title = doc.get("title", "").lower().strip()
        if url in memory.accepted_urls:
            continue
        # Skip if very similar title already seen
        too_similar = False
        for seen in seen_titles:
            if _title_similarity(title, seen) > 0.75:
                too_similar = True
                break
        if not too_similar:
            seen_titles.add(title)
            deduped.append(doc)
    return deduped


def _title_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity for titles."""
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    intersection = wa & wb
    return len(intersection) / max(len(wa), len(wb))
