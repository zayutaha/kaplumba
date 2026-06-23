"""Document cleaning — swap in small model, clean pages one by one, swap back."""

import gc
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

CLEANER_MODEL = str(Path.home() / ".omlx" / "models" / "inf")


def _clean_one(text: str, model, tokenizer, query: str = "") -> str:
    """Single combined pass: strip HTML + filter by query relevance."""
    if query:
        sys = f'Strip HTML tags, scripts, nav menus, ads, AND any content NOT related to "{query}". Keep ONLY sentences and paragraphs relevant to "{query}". Return the exact remaining text word for word — no summaries, no paraphrasing, no truncation.'
    else:
        sys = "Strip HTML tags, scripts, nav menus, and ads from the page below. Return the EXACT readable text content word for word — no summaries, no paraphrasing, no truncation."
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": text}]
    p = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, add_special_tokens=True, enable_thinking=False)
    c = make_prompt_cache(model, None)
    s = make_sampler(0.0, 1.0)
    r = ""
    for resp in stream_generate(model, tokenizer, p, max_tokens=2048, sampler=s, prompt_cache=c,
                                 turbo_kv_bits=3, turbo_fp16_layers=2):
        r += resp.text
    return r.strip()


def normalize_docs(scraped_docs: list[dict], model, tokenizer, args,
                   query: str = "",
                   chat_template_kwargs=None) -> tuple[str, object, object]:
    """Clean all pages using a separate small model. Returns (cleaned_text, new_model, new_tokenizer).
    
    Args:
        query: If provided, only content relevant to this query is kept.
    """
    if not scraped_docs:
        return "", model, tokenizer

    # 1. Unload main model
    del model, tokenizer
    gc.collect()
    mx.clear_cache()

    # 2. Load cleaner model
    print("[INFO] Loading cleaner model...")
    cleaner, clean_tok = load(CLEANER_MODEL)

    # 3. Process each page individually
    cleaned_parts = []
    for i, doc in enumerate(scraped_docs):
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        print(f"[INFO] Cleaning page {i+1}/{len(scraped_docs)}: {title[:50]}...")
        cleaned = _clean_one(content, cleaner, clean_tok, query=query)
        cleaned_parts.append(f"## {title}\nSource: {doc.get('url', '')}\n\n{cleaned}")

    # 4. Unload cleaner model
    del cleaner, clean_tok
    gc.collect()
    mx.clear_cache()

    # 5. Reload main model
    print("[INFO] Reloading main model...")
    new_model, new_tok = load(
        str(Path(args.model).expanduser().resolve()),
        tokenizer_config={"trust_remote_code": True if args.trust_remote_code else None},
    )

    return "\n\n---\n\n".join(cleaned_parts), new_model, new_tok
