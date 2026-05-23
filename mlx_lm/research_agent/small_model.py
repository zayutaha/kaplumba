"""Small model manager — loads Qwen 2B for cheap research calls."""
import os
from pathlib import Path

from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

SMALL_MODEL_PATH = str(Path.home() / ".omlx" / "models" / "Qwen3.5-2B-MLX-9bit")


class SmallModelManager:
    """Manages a small model (Qwen 3.5 2B) for cheap inference calls."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded and self.model is not None

    def load(self, model_path: str | None = None) -> bool:
        """Load the small model. Returns True on success."""
        if self.loaded:
            return True
        path = model_path or SMALL_MODEL_PATH
        if not os.path.isdir(path):
            return False
        try:
            self.model, self.tokenizer = load(path)
            self._loaded = True
            return True
        except Exception:
            return False

    def unload(self):
        """Release the small model."""
        self.model = None
        self.tokenizer = None
        self._loaded = False
        import gc
        gc.collect()

    def call(self, messages, max_tokens, temp=0.0):
        """Generate with the small model. Returns text."""
        if not self.loaded:
            return ""

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
        cache = make_prompt_cache(self.model)
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
