"""Small model manager — loads Qwen 2B for cheap research calls."""
import gc
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .memory_manager import unload_for_small_model

SMALL_MODEL_PATH = str(Path.home() / ".omlx" / "models" / "Qwen3.5-2B-MLX-9bit")


class SmallModelManager:
    """Manages a small model (Qwen 3.5 2B) for cheap inference calls.
    
    Before loading, unloads enough layers from the main model to fit.
    """

    def __init__(self, turbo_kv_bits: int = 3):
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._restore = lambda: None
        self.turbo_kv_bits = turbo_kv_bits

    @property
    def loaded(self) -> bool:
        return self._loaded and self.model is not None

    def load(self, main_model=None, model_path: str | None = None) -> bool:
        """Load the small model. Optionally unload main model layers first."""
        if self.loaded:
            return True
        path = model_path or SMALL_MODEL_PATH
        if not os.path.isdir(path):
            return False

        # Unload main model layers to make room
        self._restore = unload_for_small_model(main_model)

        try:
            self.model, self.tokenizer = load(path)
            self._loaded = True
            return True
        except Exception:
            # Restore if load failed
            self._restore()
            return False

    def unload(self):
        """Release the small model and restore main model layers."""
        self.model = None
        self.tokenizer = None
        self._loaded = False
        gc.collect()
        if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()
        self._restore()

    def call(self, messages, max_tokens, temp=0.0):
        """Generate with the small model. Returns text."""
        if not self.loaded:
            return ""

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
        cache = make_prompt_cache(self.model, turbo_kv_bits=self.turbo_kv_bits)
        sampler = make_sampler(
            temp, top_p=1.0, top_k=0,
            xtc_special_tokens=(
                self.tokenizer.encode("\n") + list(self.tokenizer.eos_token_ids)
            ),
        )
        text = ""
        for resp in stream_generate(
            self.model, self.tokenizer, prompt,
            max_tokens=max_tokens, sampler=sampler,
            prompt_cache=cache,
        ):
            text += resp.text
        return text.strip()
