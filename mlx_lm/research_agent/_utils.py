"""Model call helper for research agent.

Supports routing cheap calls to a small model (Qwen 2B) and
expensive calls to the main loaded model.
"""

import os

from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

# Module-level small model manager (set by orchestrator)
_small_manager = None


def set_small_model(manager):
    """Set the small model manager for cheap calls."""
    global _small_manager
    _small_manager = manager


def has_small_model() -> bool:
    return _small_manager is not None and _small_manager.loaded


def call_cheap(messages, max_tokens, temp=0.0):
    """Route to small model if available, otherwise use main model."""
    if has_small_model():
        return _small_manager.call(messages, max_tokens, temp=temp)
    return ""


def call_model(messages, max_tokens, model, tokenizer, args,
               chat_template_kwargs=None, temp=0.0, mtp=None):
    """Call the main model with messages and return generated text."""
    if model is None or tokenizer is None:
        return call_cheap(messages, max_tokens, temp=temp) if has_small_model() else ""

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        add_special_tokens=True,
        **(chat_template_kwargs or {}),
    )
    cache = make_prompt_cache(
        model, args.max_kv_size,
        turbo_kv_bits=args.turbo_kv_bits,
        turbo_fp16_layers=args.turbo_fp16_layers,
    )
    sampler = make_sampler(
        temp, args.top_p, top_k=args.top_k,
        xtc_threshold=args.xtc_threshold,
        xtc_probability=args.xtc_probability,
        xtc_special_tokens=(
            tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
        ),
    )
    text = ""
    for resp in stream_generate(
        model, tokenizer, prompt,
        max_tokens=max_tokens, sampler=sampler,
        prompt_cache=cache,
        turbo_kv_bits=args.turbo_kv_bits,
        turbo_fp16_layers=args.turbo_fp16_layers,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        mtp=mtp if mtp is not None else args.mtp,
        prefill_step_size=args.prefill_step_size,
    ):
        text += resp.text
    return text.strip()
