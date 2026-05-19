#!/usr/bin/env python3
"""Quantize model with TurboQuant and test inference.

Usage:
  uv run python3 scripts/turboquant_test_gen.py --max-layers 0 --prompt "Hello" --max-tokens 30
  (--max-layers 0 = quantize ALL Linear layers)
"""

import argparse
import gc
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_flatten

from mlx_lm import load, generate
from mlx_lm.quant.turboquant_weights import (
    TurboQuantLinear,
    turboquant_quantize,
    effective_bpw,
)


def quantize_turbo_layer(mod, bits, group_dim, seed):
    if not isinstance(mod, nn.Linear):
        return mod
    if mod.weight.ndim != 2:
        return mod
    out_dims, in_dims = mod.weight.shape
    if in_dims % group_dim != 0:
        return mod

    W = mod.weight.astype(mx.float32)
    state = turboquant_quantize(W, bits=bits, group_dim=group_dim, seed=seed)
    mx.eval(state["packed"], state["norms"])

    tql = TurboQuantLinear(
        in_dims=in_dims, out_dims=out_dims,
        bits=bits, group_dim=group_dim, seed=seed,
        bias="bias" in mod,
    )
    tql.packed = state["packed"]
    tql.norms = state["norms"].astype(mx.float16)
    tql._centroids = state["centroids"]
    tql._signs = state["signs"]
    if "bias" in mod:
        tql.bias = mod.bias.astype(mx.float16)

    mx.eval(tql.packed, tql.norms)
    del W, state, mod
    mx.clear_cache()
    return tql


def count_tq(model):
    leaves = list(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    return sum(1 for _, m in leaves if isinstance(m, TurboQuantLinear))


def model_gb(model):
    total = 0
    for _, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if isinstance(m, TurboQuantLinear):
            total += m.packed.nbytes + m.norms.nbytes
            if m.bias is not None:
                total += m.bias.nbytes
        elif isinstance(m, nn.Linear):
            total += m.weight.nbytes
            if "bias" in m:
                total += m.bias.nbytes
    return total / (1024**3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(Path.home() / ".omlx/models/Qwen3.6-27B"))
    parser.add_argument("--max-layers", type=int, default=0,
                        help="Linear layers to quantize (0 = all)")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=30)
    args = parser.parse_args()

    print(f"Loading model lazily...")
    t0 = time.time()
    model, tokenizer, _ = load(args.model, lazy=True, return_config=True)
    print(f"  Skeleton: {time.time()-t0:.1f}s")

    # Collect quantizable Linear layers
    leaves = list(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    linear_leaves = [
        (p, m) for p, m in leaves
        if isinstance(m, nn.Linear) and m.weight.ndim == 2 and m.weight.shape[-1] % args.group_dim == 0
    ]
    print(f"  {len(linear_leaves)} quantizable Linear layers")

    n_target = len(linear_leaves) if args.max_layers <= 0 else min(args.max_layers, len(linear_leaves))
    target_ids = set(id(m) for _, m in linear_leaves[:n_target])
    bpw = effective_bpw(args.bits, args.group_dim)
    print(f"  Quantizing {n_target} layers ({args.bits}b gd={args.group_dim}, {bpw:.2f} bpw)...")

    quantized_count = 0

    def converter(mod):
        nonlocal quantized_count
        if id(mod) in target_ids:
            quantized_count += 1
            if quantized_count % 10 == 0 or quantized_count == 1:
                shape = mod.weight.shape
                print(f"  [{quantized_count}/{n_target}] {shape[0]}x{shape[1]}", flush=True)
            return quantize_turbo_layer(mod, args.bits, args.group_dim, args.seed)
        return mod

    t0 = time.time()
    leaves_new = tree_map(converter, model.leaf_modules(), is_leaf=nn.Module.is_module)
    model.update_modules(leaves_new)
    print(f"  Quantization done in {time.time()-t0:.1f}s")

    n_tq = count_tq(model)
    gb = model_gb(model)
    print(f"  TurboQuantLinear: {n_tq} layers, ~{gb:.2f} GB total weights")

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("Generating...", flush=True)
    t0 = time.time()
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )
    response = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
    elapsed = time.time() - t0
    tok_count = len(tokenizer.encode(response))
    print(f"Response: {response}")
    print(f"{tok_count} tokens in {elapsed:.1f}s ({tok_count/elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()
