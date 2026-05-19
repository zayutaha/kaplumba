#!/usr/bin/env python3
"""Quantize model IN-PLACE one layer at a time, then generate.

Usage:
  uv run python3 scripts/turboquant_quantize_run.py --prompt "Hello" --max-tokens 20
"""

import argparse
import gc
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load, generate
from mlx_lm.quant.turboquant_weights import (
    TurboQuantLinear,
    turboquant_quantize,
    effective_bpw,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(Path.home() / ".omlx/models/Qwen3.6-27B"))
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()

    print("Loading model lazily...")
    model, tokenizer, _ = load(args.model, lazy=True, return_config=True)

    # Collect all Linear layer paths + modules
    leaves = list(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    linear_paths = [
        (p, m) for p, m in leaves
        if isinstance(m, nn.Linear) and m.weight.ndim == 2 and m.weight.shape[-1] % args.group_dim == 0
    ]

    bpw = effective_bpw(args.bits, args.group_dim)
    total = len(linear_paths)
    print(f"Quantizing {total} layers IN-PLACE ({args.bits}b gd={args.group_dim}, {bpw:.2f} bpw)")

    t0 = time.time()
    for idx, (path, mod) in enumerate(linear_paths):
        out_dims, in_dims = mod.weight.shape
        has_bias = "bias" in mod

        # Load original weight (triggers disk read), quantize
        W = mod.weight.astype(mx.float32)
        state = turboquant_quantize(W, bits=args.bits, group_dim=args.group_dim, seed=args.seed)
        mx.eval(state["packed"], state["norms"])

        # Build TQ layer
        tql = TurboQuantLinear(
            in_dims=in_dims, out_dims=out_dims,
            bits=args.bits, group_dim=args.group_dim, seed=args.seed,
            bias=has_bias, chunk_size=256,
        )
        tql.packed = state["packed"]
        tql.norms = state["norms"].astype(mx.float16)
        tql._centroids = state["centroids"]
        tql._signs = state["signs"]
        if has_bias:
            tql.bias = mod.bias.astype(mx.float16)
        mx.eval(tql.packed, tql.norms)

        # Replace in-place — frees original weight
        model.update_modules(tree_unflatten([(path, tql)]))

        del W, state, mod
        gc.collect()
        mx.clear_cache()

        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"  [{idx+1}/{total}] {out_dims}x{in_dims}", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({total/elapsed:.1f} layers/s)")

    # Count TQ layers
    leaves2 = list(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    n_tq = sum(1 for _, m in leaves2 if isinstance(m, TurboQuantLinear))
    total_gb = sum(
        m.packed.nbytes + m.norms.nbytes + (m.bias.nbytes if m.bias is not None else 0)
        for _, m in leaves2 if isinstance(m, TurboQuantLinear)
    ) / 1e9
    print(f"  {n_tq} TQ layers, ~{total_gb:.2f} GB")

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("Generating...", flush=True)
    t0 = time.time()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, return_dict=False,
    )
    resp = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
    elapsed = time.time() - t0
    tok_count = len(tokenizer.encode(resp))
    print(f"Response: {resp}")
    print(f"{tok_count} tokens in {elapsed:.1f}s ({tok_count/elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()
