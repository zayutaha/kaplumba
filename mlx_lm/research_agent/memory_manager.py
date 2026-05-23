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
    """Find the actual layer storage by trying common model patterns.
    
    Returns (container_object, attr_name) where setattr/slice works.
    Models store layers in various places depending on architecture:
    - model.model.layers (LLaMA, Mistral, etc.)
    - model.language_model.model.layers (Qwen, Phi, etc.)
    - model.layers (some models, if not a @property)
    """
    patterns = [
        # Pattern 1: model.model.layers
        (['model', 'layers'], lambda m: getattr(getattr(m, 'model', None), 'layers', None)),
        # Pattern 2: model.language_model.model.layers
        (['language_model', 'model', 'layers'], lambda m: getattr(
            getattr(getattr(m, 'language_model', None), 'model', None), 'layers', None)),
        # Pattern 3: model.layers directly (if not @property)
        (['layers'], lambda m: getattr(m, 'layers', None) if not isinstance(
            getattr(type(m), 'layers', None), property) else None),
        # Pattern 4: model.transformer.layers
        (['transformer', 'layers'], lambda m: getattr(getattr(m, 'transformer', None), 'layers', None)),
    ]

    for pattern_name, getter in patterns:
        result = getter(model)
        if result is not None:
            # Navigate to the parent of 'layers'
            path = pattern_name
            obj = model
            for step in path[:-1]:  # Navigate to the parent
                obj = getattr(obj, step, None)
                if obj is None:
                    break
            else:
                # Verify layers is not a @property on the actual container
                if not isinstance(getattr(type(obj), 'layers', None), property):
                    return obj, 'layers'

    return None, None


def unload_for_small_model(main_model) -> callable:
    """Unload 50% of big model layers to fit the small model.
    
    Returns a restore function to call when done.
    """
    if main_model is None:
        return lambda: None

    parent, attr = _find_layers(main_model)
    if parent is not None:
        layers = getattr(parent, attr)
        if layers:
            if not hasattr(main_model, '_saved_layers'):
                main_model._saved_layers = layers[:]
            n = len(layers)
            kept = max(1, n // 2)
            setattr(parent, attr, layers[:kept])

    # Always free cache memory, even if we couldn't find layers
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
