"""Document normalization — batch process scraped docs into structured format."""

from ._utils import call_model


def normalize_docs(scraped_docs: list[dict], model, tokenizer, args,
                   chat_template_kwargs=None) -> list[dict]:
    """Batch normalize all scraped docs into structured format.
    
    One model call processes all docs. Returns list of normalized dicts.
    """
    if not scraped_docs:
        return []

    docs_text = ""
    for i, doc in enumerate(scraped_docs):
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        # Use more content so the model has enough to work with
        docs_text += f"\n--- DOC {i+1}: {title} ---\n{content[:3000]}\n"

    messages = [
        {"role": "system", "content": f"""You are a document normalizer.

Extract structured information from {len(scraped_docs)} documents about a topic.

For each document, extract:
- summary: 2-3 sentence factual summary
- key_facts: list of specific facts, dates, numbers, names
- entities: key people, places, organizations mentioned
- themes: main topics covered

Output format — one block per document:
DOC 1:
summary: ...
key_facts: [fact1, fact2, ...]
entities: [entity1, entity2, ...]
themes: [theme1, theme2, ...]

DOC 2:
...

Be factual. Do not add information not in the source. Prefer structure over prose."""},
        {"role": "user", "content": f"Topic context. Extract structured information from these documents:\n{docs_text}"},
    ]
    result = call_model(messages, max_tokens=2048, model=model,
                        tokenizer=tokenizer, args=args,
                        chat_template_kwargs=chat_template_kwargs,
                        temp=0.0, mtp=False)

    # Parse results
    normalized = []
    blocks = result.split("\nDOC ")
    for block in blocks:
        if not block.strip():
            continue
        if not block[0].isdigit():
            continue
        doc_num = block[0]
        body = block[2:].strip() if len(block) > 2 else ""

        summary = ""
        key_facts = []
        entities = []
        themes = []

        for line in body.splitlines():
            line = line.strip()
            if line.startswith("summary:"):
                summary = line[8:].strip()
            elif line.startswith("key_facts:"):
                raw = line[10:].strip()
                if raw.startswith("[") and raw.endswith("]"):
                    import json
                    try:
                        key_facts = json.loads(raw)
                    except json.JSONDecodeError:
                        key_facts = [raw.strip("[]")]
                else:
                    key_facts = [raw]
            elif line.startswith("entities:"):
                raw = line[9:].strip()
                if raw.startswith("["):
                    import json
                    try:
                        entities = json.loads(raw)
                    except json.JSONDecodeError:
                        entities = [raw.strip("[]")]
                else:
                    entities = [raw]
            elif line.startswith("themes:"):
                raw = line[7:].strip()
                if raw.startswith("["):
                    import json
                    try:
                        themes = json.loads(raw)
                    except json.JSONDecodeError:
                        themes = [raw.strip("[]")]
                else:
                    themes = [raw]

        idx = int(doc_num) - 1
        if idx < len(scraped_docs):
            normalized.append({
                "title": scraped_docs[idx].get("title", ""),
                "url": scraped_docs[idx].get("url", ""),
                "summary": summary,
                "key_facts": key_facts,
                "entities": entities,
                "themes": themes,
            })

    # Fallback: if we couldn't parse, return basic structure
    if not normalized:
        for doc in scraped_docs:
            normalized.append({
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "summary": doc.get("content", "")[:300],
                "key_facts": [],
                "entities": [],
                "themes": [],
            })

    return normalized
