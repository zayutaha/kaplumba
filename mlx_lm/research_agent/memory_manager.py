"""Memory manager — unloads big model layers to fit small model."""
import gc
import json
import os
from pathlib import Path

import mlx.core as mx

SMALL_MODEL_PATH = str(Path.home() / ".omlx" / "models" / "Qwen3.5-2B-MLX-9bit")


def _estimate_model_memory(main_model) -> float:
    """Estimate main model's active memory in GB."""
    return mx.get_active_memory() / 1e9


def _estimate_small_model_memory() -> float:
    """Estimate how much memory Qwen 2B needs (GB)."""
    config_path = os.path.join(SMALL_MODEL_PATH, "config.json")
    if not os.path.isfile(config_path):
        return 2.5  # default estimate
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        hidden = cfg.get("hidden_size", 2048)
        num_layers = cfg.get("num_hidden_layers", 28)
        # Each layer has ~4 matrices (Q, K, V, O) + 2 MLP → 6 * hidden^2 * 4 bytes (fp32) / 2 (quant)
        # Rough estimate: hidden_size * num_layers * 8 * 2 bytes / 1e9
        mem_gb = (hidden * hidden * num_layers * 8 * 2) / 1e9
        return max(1.5, mem_gb)
    except Exception:
        return 2.5


def _find_layers(model):
    """Find the layer list in a model."""
    inner = getattr(model, 'model', None)
    if inner is not None and hasattr(inner, 'layers'):
        return inner, 'layers'
    if hasattr(model, 'layers'):
        return model, 'layers'
    return None, None


def unload_for_small_model(main_model) -> callable:
    """Unload enough layers to fit Qwen 2B.
    
    Returns a restore function to call when done.
    """
    if main_model is None:
        return lambda: None

    parent, attr = _find_layers(main_model)
    if parent is None:
        return lambda: None

    layers = getattr(parent, attr)
    if not layers:
        return lambda: None

    # Save original
    if not hasattr(main_model, '_saved_layers'):
        main_model._saved_layers = layers[:]

    # Calculate how much to free
    main_mem = _estimate_model_memory(main_model)
    small_mem = _estimate_small_model_memory()
    context_mem = 0.5  # ~500MB for Qwen context
    needed = small_mem + context_mem

    n = len(layers)
    if main_mem <= needed or n <= 1:
        # Already small enough, or can't unload further
        return lambda: _restore_layers(main_model)

    # Each layer = main_mem / n
    mem_per_layer = main_mem / n
    layers_to_free = int(needed / mem_per_layer) + 1  # +1 for safety
    layers_to_keep = max(1, n - layers_to_free)
    layers_to_drop = n - layers_to_keep

    # Truncate
    setattr(parent, attr, layers[:layers_to_keep])
    gc.collect()
    if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
        mx.metal.clear_cache()

    def restore():
        _restore_layers(main_model)

    return restore


def _restore_layers(main_model):
    """Restore the original layers."""
    saved = getattr(main_model, '_saved_layers', None)
    if saved is None:
        return
    parent, attr = _find_layers(main_model)
    if parent is not None:
        setattr(parent, attr, saved)
    if hasattr(main_model, '_saved_layers'):
        del main_model._saved_layers
