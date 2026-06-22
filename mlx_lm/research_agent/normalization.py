"""Document cleaning — swap in small model, clean pages one by one, swap back."""

import gc
import hashlib
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

CLEANER_MODEL = str(Path.home() / ".omlx" / "models" / "inferencerlabs")


def _clean_one(text: str, model, tokenizer, args) -> str:
    """Run one page through the cleaner model and return cleaned text."""
    messages = [
        {"role": "system", "content": "Strip HTML tags, scripts, nav menus, and ads from the page below. Return the EXACT readable text content word for word — no summaries, no paraphrasing, no truncation. Output every sentence exactly as written."},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, add_special_tokens=True,
        enable_thinking=False,
    )
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, add_special_tokens=True,
    )
    cache = make_prompt_cache(model, None)
    sampler = make_sampler(0.0, 1.0)
    result = ""
    for resp in stream_generate(
        model, tokenizer, prompt, max_tokens=2048,
        sampler=sampler, prompt_cache=cache,
    ):
        result += resp.text
    return result.strip()


def normalize_docs(scraped_docs: list[dict], model, tokenizer, args,
                   chat_template_kwargs=None) -> tuple[str, object, object]:
    """Clean all pages using a separate small model. Returns (cleaned_text, new_model, new_tokenizer)."""
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
        cleaned = _clean_one(content, cleaner, clean_tok, args)
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
