import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_catalog
import model_lifecycle
import settings_store


class TestSettingsStore(unittest.TestCase):
    def test_normalize_model_options_applies_defaults_and_ignores_invalid_values(self):
        normalized = settings_store.normalize_model_options({
            "temp": 0.3,
            "top_p": 999,
            "mtp": False,
        })

        self.assertEqual(normalized["temp"], 0.3)
        self.assertEqual(normalized["top_p"], settings_store.DEFAULT_MODEL_OPTIONS["top_p"])
        self.assertFalse(normalized["mtp"])
        self.assertEqual(normalized["max_tokens"], settings_store.DEFAULT_MODEL_OPTIONS["max_tokens"])

class TestModelLifecycle(unittest.TestCase):
    def test_build_model_command_merges_per_model_options(self):
        options = {
            "temp": 0.7,
            "top_p": 0.8,
            "top_k": 0,
            "max_tokens": 2048,
            "max_kv_size": None,
            "turbo_kv_bits": 3.0,
            "turbo_fp16_layers": 2,
            "mtp": False,
            "prefill_step_size": 128,
        }
        with patch("model_lifecycle.load_model_configs", return_value={
            "demo-model": {"options": {"mtp": True, "max_kv_size": 4096, "prefill_step_size": 256}}
        }):
            command = model_lifecycle.build_model_command(
                "/tmp/demo-model",
                options,
                "system prompt",
            )

        self.assertIn("--model", command)
        self.assertIn("/tmp/demo-model", command)
        self.assertIn("--system-prompt", command)
        self.assertIn("system prompt", command)
        self.assertIn("--mtp", command)
        self.assertIn("--max-kv-size", command)
        self.assertIn("4096", command)
        self.assertIn("--prefill-step-size", command)
        self.assertIn("256", command)


class TestModelCatalog(unittest.TestCase):
    def test_list_models_returns_typed_sorted_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            models_dir = home / ".omlx" / "models"
            small = models_dir / "small-model"
            large = models_dir / "large-model"
            small.mkdir(parents=True)
            large.mkdir(parents=True)
            (small / "weights.bin").write_bytes(b"x" * 1024)
            (large / "weights.bin").write_bytes(b"x" * 2048)

            with patch("model_catalog.Path.home", return_value=home):
                with patch.object(model_catalog, "get_total_memory_bytes", return_value=16 * model_catalog.GIB):
                    with patch.object(model_catalog, "get_available_memory_bytes", return_value=12 * model_catalog.GIB):
                        models = model_catalog.list_models({})

        self.assertEqual([model.name for model in models], ["small-model", "large-model"])
        self.assertTrue(all(isinstance(model.capabilities.fits_memory, bool) for model in models))


if __name__ == "__main__":
    unittest.main()
