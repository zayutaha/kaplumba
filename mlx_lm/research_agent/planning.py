"""Research planning — evaluate coverage and generate targeted queries."""

import json

from ._utils import call_cheap, call_model
from .memory import ResearchMemory


def evaluate_coverage(memory: ResearchMemory, model, tokenizer, args,
                      chat_template_kwargs=None) -> dict:
    """Assess which dimensions need more coverage."""
    dims_str = "\n".join(f"- {d}: {v:.1f}" for d, v in memory.dimensions.items())
    messages = [
        {"role": "system", "content": f"""You are a research coverage evaluator.
Given topic coverage scores (0.0=none, 0.3=weak, 0.7=good, 1.0=saturated), 
estimate updated coverage after {memory.iteration} iterations.
Output JSON: {{"dimension": score, ...}}
Only output the JSON object."""},
        {"role": "user", "content": f"Topic: {memory.topic}\nType: {memory.topic_type}\nCurrent coverage:\n{dims_str}\nEntities seen: {memory.entities_seen[:10]}\nConcepts seen: {memory.concepts_seen[:10]}"},
    ]
    result = call_cheap(messages, max_tokens=256, temp=0.0)
    if not result:
        result = call_model(messages, max_tokens=256, model=model,
                            tokenizer=tokenizer, args=args,
                            chat_template_kwargs=chat_template_kwargs,
                            temp=0.0)
    try:
        # Find JSON in output
        start = result.index("{")
        end = result.rindex("}") + 1
        parsed = json.loads(result[start:end])
        # Merge with existing dimensions
        for d in memory.dimensions:
            if d in parsed:
                memory.dimensions[d] = min(1.0, max(0.0, float(parsed[d])))
    except (ValueError, json.JSONDecodeError):
        pass
    return dict(memory.dimensions)


def generate_queries(memory: ResearchMemory, model, tokenizer, args,
                     chat_template_kwargs=None) -> list[dict]:
    """Generate targeted search queries for weak dimensions.
    
    Returns list of {"query": str, "dimension": str, "mode": "exploit"|"explore"}
    """
    weak = memory.get_weak_dimensions(0.5)
    dims_str = ", ".join(weak[:5]) if weak else "all dimensions"

    messages = [
        {"role": "system", "content": f"""You are a search query generator for research.
Generate 3 targeted web search queries for the topic.

Output format — one query per line:
QUERY|dimension_name|exploit

70% exploit (deepen weak coverage), 30% explore (seek new angles).
Queries should be specific, use proper names and keywords.
Do NOT number. Do NOT explain."""},
        {"role": "user", "content": f"Topic: {memory.topic}\nType: {memory.topic_type}\nDimensions needing coverage: {dims_str}\nAlready searched: {memory.searched_queries[-5:]}"},
    ]
    result = call_cheap(messages, max_tokens=128, temp=0.3)
    if not result:
        result = call_model(messages, max_tokens=128, model=model,
                            tokenizer=tokenizer, args=args,
                            chat_template_kwargs=chat_template_kwargs,
                            temp=0.3)
    queries = []
    for line in result.splitlines():
        line = line.strip().lstrip("0123456789.)- ")
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            queries.append({
                "query": parts[0],
                "dimension": parts[1] if len(parts) > 1 else (weak[0] if weak else "overview"),
                "mode": parts[2] if len(parts) > 2 else "exploit",
            })
        elif line and len(line) > 5:
            queries.append({
                "query": line,
                "dimension": weak[0] if weak else "overview",
                "mode": "exploit",
            })
    return queries[:3]


def should_stop(memory: ResearchMemory, coverage: dict) -> bool:
    """Check if research should stop."""
    if memory.iteration >= 5:
        return True
    avg = sum(coverage.values()) / len(coverage) if coverage else 0
    if avg > 0.8:
        return True
    if len(memory.novelty_history) >= 2:
        if all(n < 0.1 for n in memory.novelty_history[-2:]):
            return True
    return False
