# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import os
import select
import sys
from typing import Generator, List, Optional, Union

import mlx.core as mx

from .generate import GenerationResponse, stream_generate
from .models.cache import make_prompt_cache
from .sample_utils import make_sampler
from .utils import load, sharded_load

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_PROMPT_MARKER = ">> "

PERSONALITIES: dict[str, str] = {
    "default": """Answer in as few words as needed. No preamble, no disclaimers, no filler. If unsure, say "I don't know" and stop. Be direct. Swear if it fits. Never mention being an AI.""",
    "doctor": """Explain medical stuff like you're a paramedic in a bar. Direct, practical, no bullshit. Ask what matters, tell them what to watch for, and say when they need to see a real doctor. No AI talk. No padding. Swear if the situation warrants it.""",
    "historian": """Tell history like you're recounting it to a friend over drinks. Focus on the people, the decisions, the luck, and the fuck-ups. Big themes, not just dates. Analogies to now are fine if they land. No "objectively speaking" or "it's complicated" cop-outs.""",
}


def chat(
    model,
    tokenizer,
    messages: List[dict],
    *,
    tokens: Optional[Union[List[int], "mx.array"]] = None,
    max_tokens: int = 256,
    temp: float = DEFAULT_TEMP,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    xtc_probability: float = DEFAULT_XTC_PROBABILITY,
    xtc_threshold: float = DEFAULT_XTC_THRESHOLD,
    sampler=None,
    prompt_cache=None,
    max_kv_size: Optional[int] = None,
    turbo_kv_bits: Optional[int] = None,
    turbo_fp16_layers: int = 1,
    mtp: bool = False,
    enable_thinking: bool = False,
    chat_template_kwargs: Optional[dict] = None,
) -> str:
    """Generate a chat response from the model.

    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use.
        messages (List[dict]): A list of message dictionaries with 'role' and 'content' keys.
            Example: [{"role": "user", "content": "Hello!"}]
        tokens (Optional[Union[List[int], mx.array]]): Pre-tokenized input tokens.
            If provided, this takes precedence over messages and the chat template
            is not applied. Use this for continuing from a tokenized prompt.
        max_tokens (int): Maximum number of tokens to generate. Default: 256.
        temp (float): Sampling temperature. Default: 0.0.
        top_p (float): Nucleus sampling top-p. Default: 1.0.
        top_k (int): Top-k sampling cutoff. Default: 0.
        xtc_probability (float): XTC sampling probability. Default: 0.0.
        xtc_threshold (float): XTC threshold. Default: 0.0.
        sampler: Optional custom sampler. Default: None.
        prompt_cache: Optional pre-computed prompt cache. Default: None.
        max_kv_size (int): Maximum KV cache size. Default: None.
        turbo_kv_bits (int): TurboQuant KV cache bits. Default: None.
        turbo_fp16_layers (int): Number of FP16 layers for TurboQuant. Default: 1.
        mtp (bool): Use multi-token prediction. Default: False.
        enable_thinking (bool): Enable thinking mode for supported models. Default: False.
        chat_template_kwargs (dict): Additional kwargs for apply_chat_template. Default: None.

    Returns:
        str: The generated response text.
    """

    if tokens is not None:
        if isinstance(tokens, list):
            prompt = mx.array(tokens, mx.uint32)
        else:
            prompt = tokens
    else:
        chat_template_kwargs = chat_template_kwargs or {}
        chat_template_kwargs["enable_thinking"] = enable_thinking
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

    if sampler is None:
        sampler = make_sampler(
            temp,
            top_p,
            top_k=top_k,
            xtc_threshold=xtc_threshold,
            xtc_probability=xtc_probability,
            xtc_special_tokens=(
                tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
            ),
        )

    if prompt_cache is None:
        prompt_cache = make_prompt_cache(
            model,
            max_kv_size,
            turbo_kv_bits=turbo_kv_bits,
            turbo_fp16_layers=turbo_fp16_layers,
        )

    text = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
        turbo_kv_bits=turbo_kv_bits,
        turbo_fp16_layers=turbo_fp16_layers,
        mtp=mtp,
    ):
        text += response.text

    return text


def stream_chat(
    model,
    tokenizer,
    messages: List[dict],
    *,
    tokens: Optional[Union[List[int], "mx.array"]] = None,
    max_tokens: int = 256,
    temp: float = DEFAULT_TEMP,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    xtc_probability: float = DEFAULT_XTC_PROBABILITY,
    xtc_threshold: float = DEFAULT_XTC_THRESHOLD,
    sampler=None,
    prompt_cache=None,
    max_kv_size: Optional[int] = None,
    turbo_kv_bits: Optional[int] = None,
    turbo_fp16_layers: int = 1,
    mtp: bool = False,
    enable_thinking: bool = False,
    chat_template_kwargs: Optional[dict] = None,
) -> Generator[GenerationResponse, None, None]:
    """Stream chat responses from the model.

    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use.
        messages (List[dict]): A list of message dictionaries with 'role' and 'content' keys.
        tokens (Optional[Union[List[int], mx.array]]): Pre-tokenized input tokens.
            If provided, this takes precedence over messages.
        max_tokens (int): Maximum number of tokens to generate. Default: 256.
        temp (float): Sampling temperature. Default: 0.0.
        top_p (float): Nucleus sampling top-p. Default: 1.0.
        top_k (int): Top-k sampling cutoff. Default: 0.
        xtc_probability (float): XTC sampling probability. Default: 0.0.
        xtc_threshold (float): XTC threshold. Default: 0.0.
        sampler: Optional custom sampler. Default: None.
        prompt_cache: Optional pre-computed prompt cache. Default: None.
        max_kv_size (int): Maximum KV cache size. Default: None.
        turbo_kv_bits (int): TurboQuant KV cache bits. Default: None.
        turbo_fp16_layers (int): Number of FP16 layers for TurboQuant. Default: 1.
        mtp (bool): Use multi-token prediction. Default: False.
        enable_thinking (bool): Enable thinking mode for supported models. Default: False.
        chat_template_kwargs (dict): Additional kwargs for apply_chat_template. Default: None.

    Yields:
        GenerationResponse: The generated response with metadata.
    """

    if tokens is not None:
        if isinstance(tokens, list):
            prompt = mx.array(tokens, mx.uint32)
        else:
            prompt = tokens
    else:
        chat_template_kwargs = chat_template_kwargs or {}
        chat_template_kwargs["enable_thinking"] = enable_thinking
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

    if sampler is None:
        sampler = make_sampler(
            temp,
            top_p,
            top_k=top_k,
            xtc_threshold=xtc_threshold,
            xtc_probability=xtc_probability,
            xtc_special_tokens=(
                tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
            ),
        )

    if prompt_cache is None:
        prompt_cache = make_prompt_cache(
            model,
            max_kv_size,
            turbo_kv_bits=turbo_kv_bits,
            turbo_fp16_layers=turbo_fp16_layers,
        )

    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
        turbo_kv_bits=turbo_kv_bits,
        turbo_fp16_layers=turbo_fp16_layers,
        mtp=mtp,
    ):
        yield response


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Sampling top-k"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0,
        help="Thresold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--turbo-kv-bits",
        type=int,
        default=None,
        help="TurboQuant KV cache compression bits (1-4). "
        "3-bit gives 4.6x compression. Default: no compression.",
    )
    parser.add_argument(
        "--turbo-fp16-layers",
        type=int,
        default=1,
        help="Number of first/last layers to keep in FP16 "
        "when using --turbo-kv-bits. Default: 1.",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Number of bits for KV cache quantization. "
        "Defaults to no quantization.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization. Default: 64.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=5000,
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards. Default: 5000.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Step size for prompt prefill processing. "
        "Larger values process more tokens per forward pass "
        "but use more memory. Default: 2048.",
    )
    parser.add_argument(
        "--mtp",
        action="store_true",
        help="Use native Multi-Token Prediction for speculative decoding",
    )
    parser.add_argument(
        "--chat-template-args",
        type=str,
        default=None,
        help="JSON string of arguments for tokenizer's apply_chat_template, e.g. '{\"enable_thinking\":false}'",
    )
    parser.add_argument(
        "--tokens",
        "-t",
        type=str,
        default=None,
        help="Pre-tokenized input tokens (comma-separated integers). If provided, takes precedence over --prompt/--message and chat template is not applied.",
    )
    parser.add_argument(
        "--prompt-marker",
        type=str,
        default=DEFAULT_PROMPT_MARKER,
        help="Prompt marker used for interactive input.",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    mx.random.seed(args.seed)

    if group.size() > 1:
        if args.adapter_path:
            parser.error("Adapters not supported in distributed mode")
        model, tokenizer = sharded_load(args.model, pipeline_group, tensor_group)
    else:
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter_path,
            tokenizer_config={
                "trust_remote_code": True if args.trust_remote_code else None
            },
        )

    def print_help():
        rprint("The command list:")
        rprint("- 'q' to exit")
        rprint("- 'r' to reset the chat")
        rprint("- '/clear' to clear the conversation")
        rprint("- 'h' to display these commands")
        rprint("- '/think <message>' to enable thinking mode for that message")
        rprint(f"- '/personality_set <name>' to change personality (available: {', '.join(PERSONALITIES.keys())})")
        rprint("- '/search <query>' to search the web and generate a response")
        rprint("- '/memory' to show current GPU memory usage")
        rprint("- '/unload <pct>' to unload N% of model layers (not yet implemented)")
        rprint("- '/reload' to reload all previously unloaded layers (not yet implemented)")

    rprint(f"[INFO] Starting chat session with {args.model}.")
    print_help()

    # Make system_prompt mutable for dynamic personality switching
    current_system_prompt = args.system_prompt

    chat_template_kwargs = {}
    if args.chat_template_args:
        chat_template_kwargs = json.loads(args.chat_template_args)

    if args.tokens is not None:
        prompt = mx.array([int(t) for t in args.tokens.split(",")], mx.uint32)
        rprint(f"[INFO] Using provided tokens: {prompt.tolist()[:10]}...")
    else:
        prompt = None

    prompt_cache = make_prompt_cache(
        model,
        args.max_kv_size,
        turbo_kv_bits=args.turbo_kv_bits,
        turbo_fp16_layers=args.turbo_fp16_layers,
    )

    message_history: list = []

    while True:
        if prompt is None:
            try:
                query = input(args.prompt_marker if rank == 0 else "")
            except EOFError:
                rprint("\n[INFO] Exiting...")
                break
            if query == "q":
                break
            if query == "r":
                prompt_cache = make_prompt_cache(
                    model,
                    args.max_kv_size,
                    turbo_kv_bits=args.turbo_kv_bits,
                    turbo_fp16_layers=args.turbo_fp16_layers,
                )
                message_history.clear()
                continue
            if query == "/clear":
                prompt_cache = make_prompt_cache(
                    model,
                    args.max_kv_size,
                    turbo_kv_bits=args.turbo_kv_bits,
                    turbo_fp16_layers=args.turbo_fp16_layers,
                )
                message_history.clear()
                rprint("[INFO] Conversation cleared.")
                continue
            if query == "h":
                print_help()
                continue
            if query.startswith("/personality_set "):
                # Extract the personality name
                personality_name = query[len("/personality_set "):].strip()
                if personality_name in PERSONALITIES:
                    current_system_prompt = PERSONALITIES[personality_name]
                    prompt_cache = make_prompt_cache(
                        model,
                        args.max_kv_size,
                        turbo_kv_bits=args.turbo_kv_bits,
                        turbo_fp16_layers=args.turbo_fp16_layers,
                    )
                    message_history.clear()
                    rprint(f"[INFO] Personality set to '{personality_name}'.")
                else:
                    available = ", ".join(PERSONALITIES.keys())
                    rprint(f"[ERROR] Unknown personality. Available: {available}")
                continue
            
            # Handle /memory command - show GPU memory usage
            if query == "/memory":
                rprint("[INFO] Memory stats")
                cache_mem = mx.get_cache_memory() / 1e9
                peak_mem = mx.get_peak_memory() / 1e9
                rprint(f"[INFO] Cache memory: {cache_mem:.2f} GB | Peak memory: {peak_mem:.2f} GB")
                continue
            
            # Handle /search command - web search with scraping
            if query.startswith("/search "):
                search_query = query[8:].strip()
                if not search_query:
                    rprint("[ERROR] /search requires a query. Usage: /search <query>")
                    continue
                try:
                    from .web_search import search_web, scrape_url
                    rprint("[INFO] Searching web...")
                    results = search_web(search_query, num_results=5)
                    if not results:
                        rprint("[INFO] No results found.")
                        continue
                    # Scrape ALL results for full context
                    rprint(f"[INFO] Scraping {len(results)} pages...")
                    search_context = ""
                    for result in results:
                        if result.get("url"):
                            scraped = scrape_url(result["url"])
                            if scraped:
                                search_context += f"Full content from {result['url']}:\n{scraped}\n\n---\n\n"
                    
                    # Create a message with search context
                    messages = []
                    if current_system_prompt is not None:
                        messages.append({"role": "system", "content": current_system_prompt})
                    messages.append({"role": "user", "content": f"""You are a search results processor. Your job has two steps:

Step 1 — CLEAN: Remove all navigation boilerplate, reference sections, category lists, "see also" lists, table-of-contents, "retrieved from" footers, and any other non-content text from the search results below. Keep only the substantive article text.

Step 2 — ANSWER: Based on the cleaned information, provide a clear, concise answer to the query. Cite sources by name.

Query: {search_query}

Raw search results:
{search_context}"""})
                    
                    # Log search context to file
                    import datetime as _dt
                    _logpath = f"/tmp/mlx_search_{_dt.datetime.now():%Y%m%d_%H%M%S}.log"
                    with open(_logpath, "w") as _f:
                        _f.write(messages[-1]["content"])
                    rprint(f"[INFO] Search context logged to {_logpath}")

                    message_history.append({"role": "user", "content": search_query})
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        add_special_tokens=True,
                        **chat_template_kwargs,
                    )
                    rprint("[INFO] Generating response from search results...\n")
                    continue  # Skip to generation with search-augmented prompt
                except Exception as e:
                    rprint(f"[ERROR] Search failed: {str(e)}")
                    continue
            elif query.startswith("/unload "):
                # Parse unload percentage
                try:
                    pct_str = query[8:].strip()
                    unload_pct = int(pct_str)
                    if not (0 <= unload_pct <= 100):
                        rprint("[ERROR] Unload percentage must be between 0 and 100")
                        continue
                    rprint(f"[INFO] Unloading {unload_pct}% of model layers not yet implemented.")
                    # TODO: Implement layer unloading when MLX supports it
                except ValueError:
                    rprint("[ERROR] Usage: /unload <percentage>")
                continue
            elif query == "/reload":
                rprint("[INFO] Reloading model layers not yet implemented.")
                # TODO: Implement layer reloading when MLX supports it
                continue
            
            # Check for /think prefix to enable thinking for this message
            thinking_kwargs = dict(chat_template_kwargs)
            used_thinking = False
            if query.startswith("/think"):
                query = query[6:].lstrip()
                thinking_kwargs["enable_thinking"] = True
                used_thinking = True
            else:
                thinking_kwargs["enable_thinking"] = False

            # Multi-turn KV cache continuation:
            # First turn — tokenize full conversation (system + user query).
            # Subsequent turns — system prompt and prior conversation are
            # already in the KV cache at correct RoPE positions. Only
            # tokenize the new user message to eliminate redundant prefill
            # of the full conversation history each turn.
            is_first_turn = not message_history
            if is_first_turn:
                messages = []
                if current_system_prompt is not None:
                    messages.append({"role": "system", "content": current_system_prompt})
                messages.append({"role": "user", "content": query})
            else:
                messages = [{"role": "user", "content": query}]

            message_history.append({"role": "user", "content": query})
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                add_special_tokens=is_first_turn,
                **thinking_kwargs,
            )

        last_response = None
        stop_generation = False
        response_text = ""
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(
                args.temp,
                args.top_p,
                top_k=args.top_k,
                xtc_threshold=args.xtc_threshold,
                xtc_probability=args.xtc_probability,
                xtc_special_tokens=(
                    tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
                ),
            ),
            prompt_cache=prompt_cache,
            turbo_kv_bits=args.turbo_kv_bits,
            turbo_fp16_layers=args.turbo_fp16_layers,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
            mtp=args.mtp,
            prefill_step_size=args.prefill_step_size,
        ):
            response_text += response.text
            rprint(response.text, flush=True, end="")
            last_response = response
            if sys.platform != "win32":
                if select.select([sys.stdin], [], [], 0)[0]:
                    try:
                        char = os.read(sys.stdin.fileno(), 1)
                        if char == b"\x04" or char == b"":
                            rprint("\n[INFO] Generation stopped by user.")
                            stop_generation = True
                            break
                    except Exception:
                        pass
        if not stop_generation:
            rprint()
        if last_response and not stop_generation:
            message_history.append({"role": "assistant", "content": response_text})
            rprint(
                f"[INFO] Generated {last_response.generation_tokens} tokens "
                f"at {last_response.generation_tps:.2f} tokens/sec "
                f"(peak memory: {last_response.peak_memory:.2f} GB)"
            )
            
            # If thinking was used, reset the prompt cache to avoid state leakage
            if used_thinking:
                prompt_cache = make_prompt_cache(
                    model,
                    args.max_kv_size,
                    turbo_kv_bits=args.turbo_kv_bits,
                    turbo_fp16_layers=args.turbo_fp16_layers,
                )
                rprint("[INFO] Thinking cache cleared.")

        prompt = None
        if stop_generation:
            rprint("[INFO] Press Ctrl-d again to exit or enter a new message.")


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
