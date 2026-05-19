#!/usr/bin/env python3
"""Validate TurboQuant for weights — loads ONE safetensor shard at a time.

Usage:
  uv run python3 scripts/turboquant_validate_weights.py

Processes each 2D weight tensor one-at-a-time by reading individual
safetensor shard files. Never loads more than one shard at once.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.quant.turboquant_weights import (
    turboquant_quantize,
    turboquant_dequantize,
    effective_bpw,
    affine_effective_bpw,
    cosine_similarity,
)

# (tq_bits, tq_group_dim, aff_bits, aff_group_size, label)
CONFIGS = [
    (4, 128, 4, 64,   "TQ 4b d128 vs Affine 4b g64"),
    (4, 128, 4, 128,  "TQ 4b d128 vs Affine 4b g128"),
    (3, 128, 3, 64,   "TQ 3b d128 vs Affine 3b g64"),
    (3, 128, 4, 64,   "TQ 3b d128 vs Affine 4b g64"),
    (3, 64,  4, 64,   "TQ 3b d64  vs Affine 4b g64"),
    (2, 128, 4, 64,   "TQ 2b d128 vs Affine 4b g64"),
]

# Only test weight names containing these substrings
TARGET_KEYS = [
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
]


def load_weight(model_path, weight_name, weight_map):
    """Load a single weight tensor from its safetensor shard."""
    shard_file = weight_map[weight_name]
    shard_path = model_path / shard_file
    tensors = mx.load(str(shard_path))
    w = tensors[weight_name]
    mx.eval(w)
    return w


def mse(a, b):
    return ((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).mean()


def max_err(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))


def process_weight(weight_name, W, results):
    O, I = W.shape
    print(f"\n  [{weight_name}] {O}x{I} {W.dtype}")

    for tq_bits, tq_gd, aff_bits, aff_gs, label in CONFIGS:
        if I % tq_gd != 0 or I % aff_gs != 0:
            continue

        tq_bpw = effective_bpw(tq_bits, tq_gd)
        aff_bpw = affine_effective_bpw(aff_bits, aff_gs)

        W_f32 = W.astype(mx.float32)

        # --- TurboQuant ---
        t0 = time.time()
        state = turboquant_quantize(W_f32, bits=tq_bits, group_dim=tq_gd)
        W_tq = turboquant_dequantize(state)
        mx.eval(W_tq)
        t_tq = time.time() - t0
        del state
        mx.clear_cache()

        mse_tq = mse(W_f32, W_tq)
        cos_tq = cosine_similarity(W_f32, W_tq)
        max_tq = max_err(W_f32, W_tq)
        del W_tq
        mx.clear_cache()

        # --- Affine ---
        t0 = time.time()
        W_a, s, b = mx.quantize(W_f32, bits=aff_bits, group_size=aff_gs)
        W_aff = mx.dequantize(W_a, s, b, bits=aff_bits, group_size=aff_gs)
        mx.eval(W_aff)
        t_aff = time.time() - t0
        del W_a, s, b
        mx.clear_cache()

        mse_aff = mse(W_f32, W_aff)
        cos_aff = cosine_similarity(W_f32, W_aff)
        max_aff = max_err(W_f32, W_aff)
        del W_aff
        mx.clear_cache()

        win = "TQ" if mse_tq.item() < mse_aff.item() else "AFF"

        row = {
            "weight": weight_name,
            "shape": f"{O}x{I}",
            "label": label,
            "tq_bpw": tq_bpw, "aff_bpw": aff_bpw,
            "tq_mse": mse_tq.item(), "aff_mse": mse_aff.item(),
            "tq_cos": cos_tq.item(), "aff_cos": cos_aff.item(),
            "tq_maxerr": max_tq.item(), "aff_maxerr": max_aff.item(),
            "tq_time_s": t_tq, "aff_time_s": t_aff,
            "winner": win,
        }
        results.append(row)

        print(f"    {label:30s} | TQ {tq_bpw:.2f}bpw MSE={mse_tq.item():.2e} cos={cos_tq.item():.6f} | "
              f"Aff {aff_bpw:.2f}bpw MSE={mse_aff.item():.2e} cos={cos_aff.item():.6f} | {win}")

    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(Path.home() / ".omlx/models/Qwen3.6-27B"))
    parser.add_argument("--max-layers", type=int, default=3,
                        help="Process at most this many DecoderLayer blocks. -1 = all.")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices, e.g. '0,1,3'. Overrides --max-layers.")
    args = parser.parse_args()

    model_path = Path(args.model)

    # Read safetensors index
    with open(model_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Collect matching weight names grouped by layer
    layer_weights = {}
    for wname in weight_map:
        if not any(t in wname for t in TARGET_KEYS):
            continue
        # Parse layer index from name like model.language_model.layers.3.self_attn.q_proj.weight
        parts = wname.split(".")
        try:
            idx = int(parts[3])  # "layers.{idx}"
        except (ValueError, IndexError):
            continue
        layer_weights.setdefault(idx, []).append(wname)

    print(f"Found {len(layer_weights)} layers with Linear weights.")

    # Determine which layers to process
    if args.layers:
        layer_idxs = [int(x.strip()) for x in args.layers.split(",")]
    elif args.max_layers > 0:
        layer_idxs = sorted(layer_weights.keys())[:args.max_layers]
    else:
        layer_idxs = sorted(layer_weights.keys())

    print(f"Processing layers: {layer_idxs} ({len(layer_idxs)} blocks)")

    all_results = []

    for lidx in layer_idxs:
        weight_names = layer_weights[lidx]
        print(f"\n=== Layer {lidx} ({len(weight_names)} weights) ===")

        for wname in sorted(weight_names):
            # Load ONE tensor at a time
            W = load_weight(model_path, wname, weight_map)
            process_weight(wname, W, all_results)
            del W
            gc.collect()
            mx.clear_cache()

    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    if not all_results:
        print("No results.")
        return

    config_labels = sorted(set(r["label"] for r in all_results))
    for label in config_labels:
        rows = [r for r in all_results if r["label"] == label]
        avg_tq_mse = np.mean([r["tq_mse"] for r in rows])
        avg_aff_mse = np.mean([r["aff_mse"] for r in rows])
        avg_tq_cos = np.mean([r["tq_cos"] for r in rows])
        avg_aff_cos = np.mean([r["aff_cos"] for r in rows])
        wins = sum(1 for r in rows if r["winner"] == "TQ")
        total = len(rows)
        print(f"  {label:30s} | "
              f"TQ MSE={avg_tq_mse:.2e} cos={avg_tq_cos:.6f} | "
              f"Aff MSE={avg_aff_mse:.2e} cos={avg_aff_cos:.6f} | "
              f"TQ wins {wins}/{total}")


if __name__ == "__main__":
    main()
