import unittest
import shutil
import os
import mlx.core as mx
from pathlib import Path
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
from mlx_lm.chat import DEFAULT_MAX_TOKENS

class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.model_path = str(Path.home() / ".omlx" / "models" / "Qwwwon")
        self.chats_dir = Path("chats")
        self.chats_dir.mkdir(exist_ok=True)
        self.test_name = "test_persistence_flow"
        
        # Load model/tokenizer once for tests
        print(f"Loading model from {self.model_path}")
        self.model, self.tokenizer = load(self.model_path)
        
    def tearDown(self):
        # Clean up files
        for f in ["test_persistence_flow.json", "test_persistence_flow.safetensors"]:
            if (self.chats_dir / f).exists():
                os.remove(self.chats_dir / f)

    def test_save_load_flow(self):
        # 1. Simulate initial conversation
        prompt_cache = make_prompt_cache(self.model, max_kv_size=None)
        
        # 2. Save
        cache_path = self.chats_dir / f"{self.test_name}.safetensors"
        json_path = self.chats_dir / f"{self.test_name}.json"
        
        mx.eval(prompt_cache)
        save_prompt_cache(str(cache_path), prompt_cache, metadata={"chat_name": self.test_name})
        
        import json
        with open(json_path, "w") as f:
            json.dump({
                "message_history": [],
                "model_type": getattr(self.model, "model_type", "unknown"),
                "mtp_enabled": False
            }, f)
            
        # 3. Reload
        print("Reloading with warm-up...")
        loaded_cache = load_prompt_cache(str(cache_path))
        mx.eval(loaded_cache)
        
        # Warm-up pass
        dummy_tokens = mx.array([[self.tokenizer.bos_token_id or 1]], dtype=mx.uint32)
        self.model(dummy_tokens, cache=loaded_cache)
        mx.eval(loaded_cache)
        print("Load and warm-up success.")

    def test_save_load_turboquant_flow(self):
        # 1. Simulate initial conversation with TurboQuant
        # Force a small model to use TurboQuant for KVCache layers
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache
        prompt_cache = make_prompt_cache(self.model, max_kv_size=None, turbo_kv_bits=4)
        
        # Verify at least one layer is TurboQuant
        self.assertTrue(any(isinstance(c, TurboQuantKVCache) for c in prompt_cache))
        
        # 2. Save
        cache_path = self.chats_dir / f"{self.test_name}_tq.safetensors"
        json_path = self.chats_dir / f"{self.test_name}_tq.json"
        
        mx.eval(prompt_cache)
        save_prompt_cache(str(cache_path), prompt_cache, metadata={"chat_name": self.test_name})
        
        import json
        with open(json_path, "w") as f:
            json.dump({
                "message_history": [],
                "model_type": getattr(self.model, "model_type", "unknown"),
                "mtp_enabled": False
            }, f)
            
        # 3. Reload
        print("Reloading TurboQuant cache with warm-up...")
        loaded_cache = load_prompt_cache(str(cache_path))
        mx.eval(loaded_cache)
        
        # Warm-up pass
        dummy_tokens = mx.array([[self.tokenizer.bos_token_id or 1]], dtype=mx.uint32)
        # TurboQuant caches need codecs built before use
        for c in loaded_cache:
            if isinstance(c, TurboQuantKVCache):
                # We need dummy data to build codecs if they weren't restored
                c.update_and_fetch(mx.zeros((1, 16, 1, 128)), mx.zeros((1, 16, 1, 128)))
        
        self.model(dummy_tokens, cache=loaded_cache)
        mx.eval(loaded_cache)
        print("TurboQuant load and warm-up success.")

        self.assertEqual(len(loaded_cache), len(prompt_cache))

if __name__ == "__main__":
    unittest.main()
