# Copyright © 2023-2024 Apple Inc.

import argparse
import gc
import json
import signal
from typing import Generator, List, Optional, Union

import mlx.core as mx

if __name__ == "__main__" and __package__ is None:
    __package__ = "mlx_lm"

from .generate import GenerationResponse, stream_generate
from mlx_lm.models.cache import make_prompt_cache
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

MATH_LATEX_PROMPT = """MATH AND LATEX OUTPUT

When writing mathematical expressions or scientific notation, use standard LaTeX within $...$ for inline math and $$...$$ for display math. Examples:

- Inline: $x^2 + y^2 = z^2$ or $E = mc^2$
- Fractions: $\\frac{a}{b}$
- Greek: $\\alpha$, $\\beta$, $\\gamma$, $\\theta$
- Sum/Prod: $\\sum_{i=1}^n$, $\\prod_{k=1}^n$
- Integrals: $\\int_a^b f(x) dx$, $\\oint$
- Limits: $\\lim_{x \\to \\infty}$
- Sub/superscript with braces: $x_{i}$, $e^{i\\pi}$
- Matrices: $\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$
- Cases: $\\begin{cases} x & \\text{if } y \\\\ z & \\text{otherwise} \\end{cases}$
- Align: $\\begin{aligned} x &= y \\\\ &= z \\end{aligned}$
- Arrows: $\\to$, $\\implies$, $\\iff$
- Sets: $\\in$, $\\notin$, $\\subset$, $\\cup$, $\\cap$
- Always use braces for subscripts/superscripts: $x_{i}$ not $x_i$, $e^{i\\pi}$ not $e^i\\pi$
- Use \\text{...} for text inside math: $\\text{if } x > 0$
"""

from textual_ui.personas import PERSONALITIES


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
        default=MATH_LATEX_PROMPT,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--turbo-kv-bits",
        type=float,
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
        default=128,
        help="Step size for prompt prefill processing. "
        "Larger values process more tokens per forward pass "
        "but use more memory. Default: 256.",
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
    parser.add_argument(
        "--mfa",
        action="store_true",
        help="Use mlx-mfa STEEL kernel for optimized attention (causal prefill only).",
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

    if args.mfa:
        try:
            from mlx_mfa.integrations.mlx_lm import patch_mlx_lm
            patch_mlx_lm(verbose=True)
        except ImportError:
            print("[WARNING] mlx-mfa not installed.")
        except Exception as e:
            print(f"[WARNING] mlx-mfa patch failed: {e}")

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
        rprint("- '/research <topic>' to research a topic in-depth (8 pages, detailed report)")
        rprint("- '/memory' to show current GPU memory usage")
        rprint("- '/unload <pct>' to unload N% of model layers")
        rprint("- '/reload' to reload all previously unloaded layers")
        rprint("- '/mtp' to toggle multi-token prediction on/off")

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
    _cache_stale = False

    # SIGINT handler for TUI interrupts — stays active permanently
    _interrupted = [False]

    def _sigint_handler(s, f):
        _interrupted[0] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    while True:
        if _interrupted[0]:
            _interrupted[0] = False
            rprint("\n[INFO] Generation stopped.")
            if prompt is not None:
                continue
        if prompt is None:
            try:
                query = input(args.prompt_marker if rank == 0 else "")
            except (EOFError, KeyboardInterrupt):
                if _interrupted[0]:
                    _interrupted[0] = False
                    continue
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
            
            # Handle /search — quick web answer
            if query.startswith("/search "):
                search_query = query[8:].strip()
                if not search_query:
                    rprint("[ERROR] Usage: /search <query>")
                    continue
                try:
                    from .web_search import search_web, scrape_url, is_relevant

                    # Generate 3 queries
                    rprint("[INFO] Generating search queries...")
                    qgen_messages = [
                        {"role": "system", "content": "You are a search query generator. Given a question, output 3 concise web search queries on separate lines. Each query must cover a different angle. Use proper names and keywords. Do NOT number. Do NOT explain.\n\nExample:\nQuestion: what happened to elon musk?\nOutput:\nelon musk news 2026\nelon musk latest updates\nelon musk biography history"},
                        {"role": "user", "content": f"Question: {search_query}\nOutput:"},
                    ]
                    qgen_prompt = tokenizer.apply_chat_template(
                        qgen_messages, add_generation_prompt=True, add_special_tokens=True, **chat_template_kwargs,
                    )
                    qgen_cache = make_prompt_cache(model, args.max_kv_size, turbo_kv_bits=args.turbo_kv_bits, turbo_fp16_layers=args.turbo_fp16_layers)
                    qgen_sampler = make_sampler(args.temp, args.top_p, top_k=args.top_k, xtc_threshold=args.xtc_threshold, xtc_probability=args.xtc_probability, xtc_special_tokens=(tokenizer.encode("\n") + list(tokenizer.eos_token_ids)))
                    qgen_text = ""
                    for resp in stream_generate(model, tokenizer, qgen_prompt, max_tokens=256, sampler=qgen_sampler, prompt_cache=qgen_cache, turbo_kv_bits=args.turbo_kv_bits, turbo_fp16_layers=args.turbo_fp16_layers, kv_bits=args.kv_bits, kv_group_size=args.kv_group_size, quantized_kv_start=args.quantized_kv_start, mtp=args.mtp, prefill_step_size=args.prefill_step_size):
                        qgen_text += resp.text
                    queries = [l.strip().lstrip("0123456789.)- ") for l in qgen_text.splitlines() if l.strip() and len(l.strip()) > 3][:3]
                    if not queries:
                        queries = [search_query]
                    if search_query not in queries:
                        queries.append(search_query)

                    # Search + scrape 3 pages
                    search_context = ""
                    seen_urls = set()
                    for q in queries:
                        rprint(f"[INFO] Searching: {q}")
                        for result in search_web(q, num_results=5):
                            url = result.get("url", "")
                            title = result.get("title", "")
                            if url and url not in seen_urls and is_relevant(title, result.get("snippet", ""), search_query):
                                seen_urls.add(url)
                                rprint(f"  -> scraping: {title}")
                                scraped = scrape_url(url)
                                if scraped:
                                    search_context += f"## {title}\nSource: {url}\n\n{scraped}\n\n---\n\n"
                                break

                    if not search_context:
                        rprint("[ERROR] No content could be scraped.")
                        continue

                    messages = []
                    if current_system_prompt is not None:
                        messages.append({"role": "system", "content": current_system_prompt})
                    messages.append({"role": "user", "content": f"Based on the search results below, provide a concise answer to: {search_query}\n\n{search_context}"})

                    import datetime as _dt
                    _logpath = f"/tmp/mlx_search_{_dt.datetime.now():%Y%m%d_%H%M%S}.log"
                    with open(_logpath, "w") as _f:
                        _f.write(messages[-1]["content"])
                    rprint(f"[INFO] Context logged to {_logpath}")

                    message_history.append({"role": "user", "content": search_query})
                    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, add_special_tokens=True, **chat_template_kwargs)
                    _cache_stale = True
                    rprint("[INFO] Generating answer...\n")
                    continue
                except Exception as e:
                    rprint(f"[ERROR] Search failed: {str(e)}")
                    continue

            # Handle /research — deep context research via research_agent
            if query.startswith("/research "):
                topic = query[10:].strip()
                if not topic:
                    rprint("[ERROR] Usage: /research <topic>")
                    continue
                try:
                    from mlx_lm.research_agent.orchestrator import run_research

                    rprint(f"[INFO] Researching: {topic}")
                    result = run_research(
                        topic=topic, model=model, tokenizer=tokenizer,
                        args=args, chat_template_kwargs=chat_template_kwargs,
                    )

                    coverage_pct = int(result["coverage"].get("overview", 0) * 100) if "overview" in result["coverage"] else int(sum(result["coverage"].values()) / max(1, len(result["coverage"])) * 100)
                    rprint(f"[INFO] Research complete: {result['num_sources']} sources, coverage ~{coverage_pct}%")

                    # Build big model prompt with context
                    sys_parts = []
                    if current_system_prompt is not None:
                        sys_parts.append(current_system_prompt)
                    sys_parts.append(f"The following research context about \"{topic}\" was gathered from web sources. Use it to inform your responses throughout this conversation.")
                    messages = [{"role": "system", "content": "\n\n".join(sys_parts)}]

                    messages.append({"role": "user", "content": f"""Below is structured research material about "{topic}".

Do NOT write a report. Instead, read the material and then ask me what specific aspect I want to know about. Wait for my follow-up question before answering.

Research material:
{result['context_section']}

Read the material and then ask me what I'd like to know about {topic}."""})

                    import datetime as _dt
                    _logpath = f"/tmp/mlx_research_{_dt.datetime.now():%Y%m%d_%H%M%S}.log"
                    with open(_logpath, "w") as _f:
                        _f.write(messages[-1]["content"])
                    rprint(f"[INFO] Research context logged to {_logpath}")

                    message_history.append({"role": "user", "content": f"Research: {topic}"})
                    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, add_special_tokens=True, **chat_template_kwargs)
                    # Disable MTP — research adds massive context that makes
                    # speculative decoding overhead not worth it.
                    args.mtp = False
                    rprint("[INFO] Research loaded. Ask me anything about it.\n")
                    continue
                except Exception as e:
                    rprint(f"[ERROR] Research failed: {str(e)}")
                    continue
            elif query.startswith("/unload "):
                try:
                    unload_pct = int(query[8:].strip())
                except ValueError:
                    rprint("[ERROR] Usage: /unload <percentage>")
                    continue
                if not (0 <= unload_pct <= 100):
                    rprint("[ERROR] Unload percentage must be between 0 and 100")
                    continue

                # Find actual writable layer storage
                parent, attr = None, None
                for src_name, src_obj in [('model', model), ('language_model', getattr(model, 'language_model', None))]:
                    if src_obj is None:
                        continue
                    inner = getattr(src_obj, 'model', None)
                    if inner is not None and hasattr(inner, 'layers') and not isinstance(getattr(type(inner), 'layers', None), property):
                        parent, attr = inner, 'layers'
                        break
                if parent is None:
                    tr = getattr(model, 'transformer', None)
                    if tr is not None and hasattr(tr, 'layers') and not isinstance(getattr(type(tr), 'layers', None), property):
                        parent, attr = tr, 'layers'
                if parent is None and hasattr(model, 'layers') and not isinstance(getattr(type(model), 'layers', None), property):
                    parent, attr = model, 'layers'

                if parent is None:
                    rprint(f"[ERROR] Could not find writable layers for {type(model).__name__}")
                    continue

                layers = getattr(parent, attr)
                n = len(layers)
                if not hasattr(model, '_saved_layers'):
                    model._saved_layers = layers[:]
                to_drop = max(1, int(n * unload_pct / 100))
                kept = n - to_drop
                setattr(parent, attr, layers[:kept])
                gc.collect()
                mx.clear_cache()
                after = mx.get_active_memory() / 1e9
                rprint(f"[INFO] Unloaded {to_drop}/{n} layers ({unload_pct}%). "
                       f"Active memory: {after:.2f} GB")
                continue

            elif query == "/reload":
                saved = getattr(model, '_saved_layers', None)
                if saved is None:
                    rprint("[INFO] No layers to reload")
                    continue

                parent, attr = None, None
                for src_name, src_obj in [('model', model), ('language_model', getattr(model, 'language_model', None))]:
                    if src_obj is None:
                        continue
                    inner = getattr(src_obj, 'model', None)
                    if inner is not None and hasattr(inner, 'layers') and not isinstance(getattr(type(inner), 'layers', None), property):
                        parent, attr = inner, 'layers'
                        break
                if parent is None:
                    tr = getattr(model, 'transformer', None)
                    if tr is not None and hasattr(tr, 'layers') and not isinstance(getattr(type(tr), 'layers', None), property):
                        parent, attr = tr, 'layers'
                if parent is None and hasattr(model, 'layers') and not isinstance(getattr(type(model), 'layers', None), property):
                    parent, attr = model, 'layers'
                if parent is None:
                    rprint("[ERROR] Could not find writable model layers")
                    continue

                setattr(parent, attr, saved)
                del model._saved_layers
                rprint("[INFO] All layers restored")
            
            # Handle /mtp toggle
            if query == "/mtp":
                # Check if model has MTP support
                has_mtp = hasattr(getattr(model, 'model', None), 'mtp') or \
                          hasattr(getattr(model, 'language_model', None), 'mtp')
                if not has_mtp:
                    rprint("[INFO] This model does not support MTP")
                    continue
                args.mtp = not args.mtp
                rprint(f"[INFO] MTP {'enabled' if args.mtp else 'disabled'}")
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
            is_first_turn = not message_history or _cache_stale
            _cache_stale = False
            if is_first_turn:
                messages = []
                if current_system_prompt is not None:
                    messages.append({"role": "system", "content": current_system_prompt})
                # Include full conversation history when cache was reset
                if message_history:
                    messages.extend(message_history)
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

        # Explicitly clear any stale interrupt flag before starting generation
        _interrupted[0] = False

        def _cache_offset(cache):
            for c in cache:
                if hasattr(c, "offset"):
                    return c.offset
            return 0
        offset_before = _cache_offset(prompt_cache) if prompt_cache else 0
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
            if _interrupted[0]:
                _interrupted[0] = False
                rprint("\n[INFO] Generation stopped by user.")
                stop_generation = True
                break
        
        # Always append whatever text was generated to history, 
        # even if interrupted, to keep history and cache in sync.
        if response_text:
            message_history.append({"role": "assistant", "content": response_text})

        if not stop_generation:
            rprint()
        if last_response and not stop_generation:
            # message_history already appended above
            
            # Log research output to file
            if message_history and len(message_history) >= 2 and message_history[-2].get("content", "").startswith("Research:"):
                import datetime as _dt
                _rpath = f"/tmp/mlx_research_output_{_dt.datetime.now():%Y%m%d_%H%M%S}.log"
                try:
                    with open(_rpath, "w") as _f:
                        _f.write(response_text)
                    rprint(f"[INFO] Research output logged to {_rpath}")
                except Exception:
                    pass

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

            # If the last generation was a search or research response,
            # reset the prompt cache so the massive search context doesn't
            # pollute subsequent turns.
            if _cache_stale:
                prompt_cache = make_prompt_cache(
                    model,
                    args.max_kv_size,
                    turbo_kv_bits=args.turbo_kv_bits,
                    turbo_fp16_layers=args.turbo_fp16_layers,
                )
                _cache_stale = False

        prompt = None
        if stop_generation:
            rprint("[INFO] Generation stopped. Ready for next message.")


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
