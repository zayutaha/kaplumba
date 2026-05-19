"""TurboQuant for model weights — PolarQuant + Metal kernels.

Reuses building blocks from mlx_lm/models/turboquant_*.py:
  - rotation.py: WHT, random signs
  - packing.py: bit-packing into uint32
  - metal.py: fused Metal quantize/dequant kernels
  - cache.py: Gaussian-optimized codebooks
"""

import time as _time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.turboquant_rotation import (
    random_diagonal_sign,
    randomized_hadamard_transform,
    inverse_randomized_hadamard,
)
from mlx_lm.models.turboquant_metal import dequant_fp16
from mlx_lm.models.turboquant_cache import _compute_gaussian_codebook, _compute_boundaries
from mlx_lm.models.turboquant_packing import packed_dim as calc_packed_dim, VALS_PER_WORD, pack_indices, unpack_indices

def _quantize_vectors(vecs, signs, boundaries, centroids, bits):
    """Quantize vectors using PolarQuant (array ops, no custom Metal kernel).

    Args:
        vecs: (N, group_dim) float32
        signs: (group_dim,) float32
        boundaries: (n_centroids-1,) float32
        centroids: (n_centroids,) float32
        bits: int

    Returns:
        packed: (N, packed_dim) uint32
        norms: (N,) float32
    """
    # L2 norm per vector
    norms = mx.sqrt((vecs * vecs).sum(axis=-1, keepdims=True))
    safe = mx.maximum(norms, 1e-8)
    normalized = vecs / safe

    # Randomized Hadamard transform
    rotated = randomized_hadamard_transform(normalized, signs)  # (N, group_dim)

    # Nearest centroid (broadcast)
    # rotated[N, d], boundaries[C-1] → idx[N, d]
    expanded = rotated[:, :, None]  # (N, d, 1)
    # Compare against each boundary: (N, d) of indices in [0, C-1]
    idx = (expanded > boundaries[None, None, :]).sum(axis=-1).astype(mx.uint8)  # (N, d)

    # Pack into uint32
    packed = pack_indices(idx, bits)
    return packed, norms.squeeze(-1)


def _dequantize_vectors(packed, norms, centroids, signs, bits, group_dim):
    """Dequantize vectors from PolarQuant representation.

    Args:
        packed: (N, packed_dim) uint32
        norms: (N,) float32
        centroids: (C,) float32
        signs: (group_dim,) float32
        bits: int
        group_dim: int

    Returns:
        (N, group_dim) float16
    """
    N = norms.shape[0]
    idx = unpack_indices(packed, bits, group_dim)  # (N, group_dim) uint8

    # Look up centroids
    values = centroids[idx.astype(mx.int32)]  # (N, group_dim)

    # Inverse rotation: WHT⁻¹ then signs
    deq = inverse_randomized_hadamard(values, signs)  # (N, group_dim)

    # Rescale by norm
    deq = deq * norms[:, None]
    return deq.astype(mx.float16)


def turboquant_quantize(W, bits=3, group_dim=128, seed=42, max_rows_f32=4096):
    """Quantize weight matrix using TurboQuant (PolarQuant).

    Processes large matrices in row-chunks to bound peak float32 memory.

    Args:
        W: (O, I) float16/32 weight matrix
        bits: quantization bits per coordinate (1-4)
        group_dim: vector dimension (power of 2, must divide I, <= 256)
        seed: random seed for rotation
        max_rows_f32: max rows to convert to float32 at once

    Returns:
        dict with keys: packed, norms, centroids, signs, bits, group_dim, O, I
    """
    O, I = W.shape
    assert group_dim > 0 and (group_dim & (group_dim - 1)) == 0, \
        f"group_dim={group_dim} must be power of 2"
    assert I % group_dim == 0, \
        f"I={I} not divisible by group_dim={group_dim}"

    n_groups = I // group_dim
    pdim = calc_packed_dim(group_dim, bits)

    signs = random_diagonal_sign(group_dim, seed=seed)
    centroids = _compute_gaussian_codebook(bits)
    boundaries = _compute_boundaries(centroids)

    packed_all = mx.zeros((O, n_groups, pdim), dtype=mx.uint32)
    norms_all = mx.zeros((O, n_groups), dtype=mx.float32)

    for start in range(0, O, max_rows_f32):
        end = min(start + max_rows_f32, O)
        batch = W[start:end].astype(mx.float32)  # small: max_rows_f32 × I × 4
        vecs = batch.reshape(-1, group_dim)
        pk, nr = _quantize_vectors(vecs, signs, boundaries, centroids, bits)
        mx.eval(pk, nr)
        packed_all[start:end] = pk.reshape(end - start, n_groups, pdim)
        norms_all[start:end] = nr.reshape(end - start, n_groups)
        del batch, vecs, pk, nr
        mx.clear_cache()

    return {
        "packed": packed_all,
        "norms": norms_all,
        "centroids": centroids,
        "signs": signs,
        "bits": bits,
        "group_dim": group_dim,
        "O": O,
        "I": I,
    }


def turboquant_dequantize(state):
    """Dequantize weight matrix from TurboQuant representation.

    Args:
        state: dict from turboquant_quantize()

    Returns:
        (O, I) float16 weight matrix
    """
    packed = state["packed"]
    norms = state["norms"]
    centroids = state["centroids"]
    signs = state["signs"]
    bits = state["bits"]
    group_dim = state["group_dim"]
    O = state["O"]
    I = state["I"]

    n_groups = I // group_dim
    n_vecs = O * n_groups

    flat_packed = packed.reshape(n_vecs, -1)
    flat_norms = norms.reshape(n_vecs)

    W_deq = _dequantize_vectors(flat_packed, flat_norms, centroids, signs, bits, group_dim)
    return W_deq.reshape(O, I)


def effective_bpw(bits, group_dim):
    """Effective bits per weight for TurboQuant."""
    overhead = 16.0 / group_dim
    return bits + overhead


def affine_effective_bpw(bits, group_size):
    """Effective bits per weight for affine quantization."""
    overhead = 64.0 / group_size
    return bits + overhead


def cosine_similarity(a, b):
    """Cosine similarity between two vectors or matrices."""
    a_f = a.flatten().astype(mx.float32)
    b_f = b.flatten().astype(mx.float32)
    dot = (a_f * b_f).sum()
    norm_a = mx.sqrt((a_f * a_f).sum())
    norm_b = mx.sqrt((b_f * b_f).sum())
    return dot / (norm_a * norm_b)


class TurboQuantLinear(nn.Module):
    """Linear layer with TurboQuant (PolarQuant) compressed weights.

    Stores weights as bit-packed codebook indices + L2 norms.
    Dequantizes in chunks during forward pass to bound peak memory.

    Args:
        in_dims: input dimension
        out_dims: output dimension
        bits: quantization bits (1-4)
        group_dim: vector dimension for grouping (power of 2)
        seed: random seed for rotation
        bias: optional bias
        chunk_size: rows to dequantize at a time (lower = less peak mem)
    """

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        bits: int = 3,
        group_dim: int = 128,
        seed: int = 42,
        bias: bool = False,
        chunk_size: int = 256,
    ):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.quant_bits = bits
        self.group_dim = group_dim
        self.seed = seed
        self.chunk_size = chunk_size

        n_groups = in_dims // group_dim
        pdim = calc_packed_dim(group_dim, bits)

        self.packed = mx.zeros((out_dims, n_groups, pdim), dtype=mx.uint32)
        self.norms = mx.zeros((out_dims, n_groups), dtype=mx.float16)

        self._centroids = _compute_gaussian_codebook(bits)
        self._signs = random_diagonal_sign(group_dim, seed=seed)

        if bias:
            self.bias_val = mx.zeros((out_dims,), dtype=mx.float16)

    def _dequantize_chunk(self, start_row: int, end_row: int) -> mx.array:
        """Dequantize a subset of rows to float16 via Metal kernel (fast, no intermediates)."""
        pk = self.packed[start_row:end_row].reshape(-1, self.packed.shape[-1])
        nr = self.norms[start_row:end_row].reshape(-1)
        Wc = dequant_fp16(pk, nr, self._centroids, self._signs,
                           self.group_dim, self.quant_bits)
        return Wc.reshape(end_row - start_row, self.in_dims)

    def __call__(self, x):
        outputs = []
        cs = min(self.chunk_size, self.out_dims)
        for start in range(0, self.out_dims, cs):
            end = min(start + cs, self.out_dims)
            Wc = self._dequantize_chunk(start, end)
            outputs.append(x @ Wc.T)

        out = mx.concatenate(outputs, axis=-1)
        b = getattr(self, "bias_val", None)
        if b is not None:
            out = out + b
        return out

    def extra_repr(self):
        return f"in={self.in_dims}, out={self.out_dims}, bits={self.quant_bits}, gd={self.group_dim}, bpw={effective_bpw(self.quant_bits, self.group_dim):.2f}"


def turboquant_quantize_model(model, bits=3, group_dim=128, seed=42, chunk_size=256):
    """Replace all nn.Linear layers with TurboQuantLinear layers.

    Processes layers one-at-a-time, freeing each original weight before
    moving to the next. Stores only paths (not module refs) to avoid
    leaking memory.

    Args:
        model: nn.Module with nn.Linear layers
        bits: quantization bits
        group_dim: vector dimension for grouping
        seed: random seed for rotation
        chunk_size: rows to dequantize at a time during forward

    Returns:
        model with TurboQuantLinear layers
    """
    from mlx.utils import tree_flatten
    leaves = list(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    paths = [p for p, m in leaves if isinstance(m, nn.Linear) and m.weight.ndim == 2 and m.weight.shape[-1] % group_dim == 0]
    del leaves

    def resolve_parent(m, path_str):
        parts = path_str.split('.')
        p = m
        for k in parts[:-1]:
            p = p[int(k)] if k.isdigit() else getattr(p, k)
        return p, parts[-1]

    N = len(paths)
    print(f"[TurboQuant] Quantizing {N} Linear layers ({bits}b gd={group_dim}, {effective_bpw(bits, group_dim):.2f} bpw)...")

    for idx, path in enumerate(paths):
        parent, attr = resolve_parent(model, path)
        mod = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
        out_dims, in_dims = mod.weight.shape
        has_bias = "bias" in mod

        W = mod.weight
        state = turboquant_quantize(W, bits=bits, group_dim=group_dim, seed=seed, max_rows_f32=2048)
        mx.eval(state['packed'], state['norms'])

        tql = TurboQuantLinear(
            in_dims=in_dims, out_dims=out_dims,
            bits=bits, group_dim=group_dim, seed=seed,
            bias=has_bias, chunk_size=chunk_size,
        )
        tql.packed = state['packed']
        tql.norms = state['norms'].astype(mx.float16)
        tql._centroids = state['centroids']
        tql._signs = state['signs']
        if has_bias:
            tql.bias_val = mod.bias.astype(mx.float16)
        mx.eval(tql.packed, tql.norms)

        if attr.isdigit():
            parent[int(attr)] = tql
        else:
            setattr(parent, attr, tql)

        del W, state, mod, tql, parent
        if (idx + 1) % 50 == 0 or idx == N - 1:
            _time.sleep(0.01)
            gc_count = _gc.collect()
            mx.clear_cache()

    return model


def turboquant_get_config(model):
    """Extract TurboQuant config from model for saving."""
    config = {"turboquant": {"group_dim": None, "bits": None, "seed": None}}
    for _, mod in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if isinstance(mod, TurboQuantLinear):
            config["turboquant"] = {
                "group_dim": mod.group_dim,
                "bits": mod.quant_bits,
                "seed": mod.seed,
            }
            break
    return config


def turboquant_model_size(model):
    """Estimate model size in GB."""
    total_bytes = 0
    for _, mod in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if isinstance(mod, TurboQuantLinear):
            total_bytes += mod.packed.nbytes + mod.norms.nbytes
            if mod.bias is not None:
                total_bytes += mod.bias.nbytes
    return total_bytes / (1024**3)
