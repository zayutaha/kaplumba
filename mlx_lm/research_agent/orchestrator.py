"""Research orchestrator — runs the full research pipeline."""

from .memory import ResearchMemory
from .framing import infer_topic_type, build_dimension_map
from .planning import evaluate_coverage, generate_queries, should_stop
from .retrieval import (
    seed_search, search_queries, score_candidate,
    deduplicate_candidates, novelty_score,
)
from .normalization import normalize_docs
from mlx_lm.web_search import scrape_url


def coverage_aware_selection(memory: ResearchMemory, count: int = 8) -> list[dict]:
    """Pick best docs per dimension, then fill remaining slots."""
    selected = []
    selected_urls = set()
    dimensions = list(memory.dimensions.keys())

    # One per dimension
    scored = sorted(memory.candidate_docs, key=lambda d: d.get("score", 0), reverse=True)
    for dim in dimensions:
        for doc in scored:
            if doc.get("url") in selected_urls:
                continue
            if doc.get("dimension") == dim or not any(d.get("dimension") == dim for d in scored):
                selected.append(doc)
                selected_urls.add(doc.get("url"))
                break

    # Fill remaining slots with highest scored
    for doc in scored:
        if len(selected) >= count:
            break
        if doc.get("url") not in selected_urls:
            selected.append(doc)
            selected_urls.add(doc.get("url"))

    return selected[:count]


def run_research(topic: str, model, tokenizer, args,
                 chat_template_kwargs=None) -> dict:
    """Run the full research pipeline. Returns context package."""
    from ._utils import set_small_model
    from .small_model import SmallModelManager

    memory = ResearchMemory(topic=topic)

    # Load small model for cheap calls (unloads big model layers if needed)
    small = SmallModelManager()
    small_ok = small.load(main_model=model)
    if small_ok:
        set_small_model(small)
        import sys
        rprint = lambda *a, **kw: print(*a, **kw)
        rprint("[INFO] Small model loaded for cheap calls")
    else:
        import sys
        rprint = lambda *a, **kw: print(*a, **kw)
        rprint("[INFO] No small model — using main model for all calls")

    try:
        # 1. Topic Framing
        memory.topic_type = infer_topic_type(
            topic, model, tokenizer, args, chat_template_kwargs
        )
        memory.dimensions = build_dimension_map(memory.topic_type, topic=topic)

        # 2. Seed Search
        seed_candidates = seed_search(topic, num_results=15)
        for c in seed_candidates:
            c["score"] = 0.5
            c["novelty"] = 0.5
            c["dimension"] = "overview"
        memory.candidate_docs = seed_candidates
        for c in seed_candidates:
            if c.get("url"):
                memory.accepted_urls.add(c["url"])

        # 3. Iterative Retrieval Loop
        for i in range(5):
            memory.iteration = i + 1

            coverage = evaluate_coverage(
                memory, model, tokenizer, args, chat_template_kwargs
            )

            if should_stop(memory, coverage):
                break

            queries = generate_queries(
                memory, model, tokenizer, args, chat_template_kwargs
            )

            new_candidates = search_queries(queries, memory)

            for doc in new_candidates:
                score_candidate(doc, memory)
                if doc.get("url"):
                    memory.accepted_urls.add(doc["url"])

            deduped = deduplicate_candidates(new_candidates, memory)

            if deduped:
                avg_novelty = sum(d.get("novelty", 0) for d in deduped) / len(deduped)
                memory.novelty_history.append(avg_novelty)

            memory.candidate_docs.extend(deduped)

        # 4. Coverage-Aware Selection
        selected = coverage_aware_selection(memory, count=8)

        # 5. Scrape selected docs — filter low-quality pages
        LOW_QUALITY_DOMAINS = ["youtube.com", "pinterest.com", "facebook.com",
                               "twitter.com", "instagram.com", "tiktok.com"]
        scraped_docs = []
        for doc in selected:
            url = doc.get("url", "")
            if not url:
                continue
            # Skip low-value domains
            if any(d in url.lower() for d in LOW_QUALITY_DOMAINS):
                continue
            content = scrape_url(url)
            if content and len(content) > 200:  # Skip near-empty pages
                scraped_docs.append({
                    "title": doc.get("title", ""),
                    "url": url,
                    "content": content,
                })

        # 6. Unload small model — restore full big model before normalization
        set_small_model(None)
        small.unload()

        # 7. Batch clean using a separate small model (never both in memory)
        context_section, new_model, new_tok = normalize_docs(
            scraped_docs, model, tokenizer, args, query=topic, chat_template_kwargs=chat_template_kwargs
        )

        # 8. Log and return
        import datetime as _dt
        _logpath = f"/tmp/mlx_research_{_dt.datetime.now():%Y%m%d_%H%M%S}.log"
        with open(_logpath, "w") as _f:
            _f.write(context_section)
        print(f"[INFO] Research context ({len(context_section)} chars) logged to {_logpath}")

        return {
            "topic": topic,
            "topic_type": memory.topic_type,
            "dimensions": list(memory.dimensions.keys()),
            "coverage": dict(memory.dimensions),
            "num_sources": len(selected),
            "context_section": context_section,
            "model": new_model,
            "tokenizer": new_tok,
        }
    finally:
        # Ensure small model is always unloaded
        if small_ok:
            set_small_model(None)
            small.unload()
