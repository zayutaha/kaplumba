"""Research orchestrator — runs the full research pipeline."""

from research_agent.memory import ResearchMemory
from research_agent.framing import infer_topic_type, build_dimension_map
from research_agent.planning import evaluate_coverage, generate_queries, should_stop
from research_agent.retrieval import (
    seed_search, search_queries, score_candidate,
    deduplicate_candidates, novelty_score,
)
from research_agent.normalization import normalize_docs
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
    memory = ResearchMemory(topic=topic)

    # 1. Topic Framing
    memory.topic_type = infer_topic_type(
        topic, model, tokenizer, args, chat_template_kwargs
    )
    memory.dimensions = build_dimension_map(memory.topic_type)

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

        # Evaluate coverage
        coverage = evaluate_coverage(
            memory, model, tokenizer, args, chat_template_kwargs
        )

        if should_stop(memory, coverage):
            break

        # Generate queries
        queries = generate_queries(
            memory, model, tokenizer, args, chat_template_kwargs
        )

        # Search
        new_candidates = search_queries(queries, memory)

        # Score
        for doc in new_candidates:
            score_candidate(doc, memory)
            if doc.get("url"):
                memory.accepted_urls.add(doc["url"])

        # Deduplicate
        deduped = deduplicate_candidates(new_candidates, memory)

        # Track novelty
        if deduped:
            avg_novelty = sum(d.get("novelty", 0) for d in deduped) / len(deduped)
            memory.novelty_history.append(avg_novelty)

        # Add to pool
        memory.candidate_docs.extend(deduped)

    # 4. Coverage-Aware Selection
    selected = coverage_aware_selection(memory, count=8)

    # 5. Scrape selected docs
    scraped_docs = []
    for doc in selected:
        url = doc.get("url", "")
        if not url:
            continue
        content = scrape_url(url)
        if content:
            scraped_docs.append({
                "title": doc.get("title", ""),
                "url": url,
                "content": content,
            })

    # 6. Normalize (one batch call)
    normalized = normalize_docs(
        scraped_docs, model, tokenizer, args, chat_template_kwargs
    )

    # 7. Build context package for big model
    # The big model receives the normalized docs + a synthesis prompt
    context_section = ""
    for nd in normalized:
        context_section += f"\n## {nd['title']}\n"
        context_section += f"Source: {nd['url']}\n"
        context_section += f"Summary: {nd['summary']}\n"
        if nd.get("key_facts"):
            context_section += f"Key facts: {'; '.join(nd['key_facts'][:10])}\n"
        if nd.get("entities"):
            context_section += f"Entities: {', '.join(nd['entities'][:10])}\n"
        if nd.get("themes"):
            context_section += f"Themes: {', '.join(nd['themes'][:5])}\n"
        context_section += "\n"

    return {
        "topic": topic,
        "topic_type": memory.topic_type,
        "dimensions": list(memory.dimensions.keys()),
        "coverage": dict(memory.dimensions),
        "num_sources": len(selected),
        "context_section": context_section,
        "normalized_docs": normalized,
        "memory": memory,
    }
