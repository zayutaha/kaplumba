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
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def chat(
    model,
    tokenizer,
    messages: List[dict],
    *,
    tokens: Optional[Union[List[int], "mx.array"]] = None,
    max_tokens: int = 256,
    temp: float = DEFAULT_TEMP,
    top_p: float = DEFAULT_TOP_P,
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
    import mx

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
    import mx

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

    rprint(f"[INFO] Starting chat session with {args.model}.")
    print_help()

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

    while True:
        if prompt is None:
            try:
                query = input(">> " if rank == 0 else "")
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
                continue
            if query == "/clear":
                prompt_cache = make_prompt_cache(
                    model,
                    args.max_kv_size,
                    turbo_kv_bits=args.turbo_kv_bits,
                    turbo_fp16_layers=args.turbo_fp16_layers,
                )
                rprint("[INFO] Conversation cleared.")
                continue
            if query == "h":
                print_help()
                continue
            messages = []
            if args.system_prompt is not None:
                messages.append({"role": "system", "content": args.system_prompt})

            # Check for /think prefix to enable thinking for this message
            thinking_kwargs = dict(chat_template_kwargs)
            if query.startswith("/think"):
                query = query[6:].lstrip()
                thinking_kwargs["enable_thinking"] = True
            else:
                thinking_kwargs["enable_thinking"] = False

            messages.append({"role": "user", "content": query})
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                **thinking_kwargs,
            )

        last_response = None
        stop_generation = False
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(
                args.temp,
                args.top_p,
                xtc_threshold=args.xtc_threshold,
                xtc_probability=args.xtc_probability,
                xtc_special_tokens=(
                    tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
                ),
            ),
            prompt_cache=prompt_cache,
            turbo_kv_bits=args.turbo_kv_bits,
            turbo_fp16_layers=args.turbo_fp16_layers,
            mtp=args.mtp,
        ):
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
            rprint(
                f"[INFO] Generated {last_response.generation_tokens} tokens "
                f"at {last_response.generation_tps:.2f} tokens/sec "
                f"(peak memory: {last_response.peak_memory:.2f} GB)"
            )

        prompt = None
        if stop_generation:
            rprint("[INFO] Press Ctrl-d again to exit or enter a new message.")


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
