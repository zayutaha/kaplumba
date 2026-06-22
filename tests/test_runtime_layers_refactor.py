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

            with patch("model_catalog.get_models_dir", return_value=models_dir):
                with patch.object(model_catalog, "get_available_memory_bytes", return_value=12 * model_catalog.GIB):
                    models = model_catalog.list_models({})

        self.assertEqual([model.name for model in models], ["small-model", "large-model"])
        self.assertTrue(all(isinstance(model.capabilities.fits_memory, bool) for model in models))


class FakeModule:
    def __init__(self):
        self.parameters_list = []

    def parameters(self):
        return self.parameters_list


class FakeModel:
    def __init__(self, n_layers=6):
        self.layers = [FakeModule() for _ in range(n_layers)]
        self._unloaded_layers = None


class TestUnloadLayers(unittest.TestCase):
    def test_unload_drops_correct_number_of_layers(self):
        model = FakeModel(n_layers=10)
        parent = model
        attr = "layers"
        all_layers = getattr(parent, attr)
        n = len(all_layers)
        to_drop = max(1, int(n * 30 / 100))
        kept = n - to_drop
        model._unloaded_layers = all_layers[kept:]
        setattr(parent, attr, all_layers[:kept])

        self.assertEqual(len(getattr(parent, attr)), kept)
        self.assertEqual(len(model._unloaded_layers), to_drop)

    def test_unload_frees_no_layers_at_zero_percent(self):
        model = FakeModel(n_layers=8)
        parent, attr = model, "layers"
        all_layers = getattr(parent, attr)
        n = len(all_layers)
        to_drop = max(1, int(n * 0 / 100))
        kept = n - to_drop
        model._unloaded_layers = all_layers[kept:]
        setattr(parent, attr, all_layers[:kept])

        self.assertEqual(len(getattr(parent, attr)), n - 1)
        self.assertEqual(len(model._unloaded_layers), 1)

    def test_auto_restore_on_next_prompt(self):
        model = FakeModel(n_layers=10)
        parent, attr = model, "layers"
        all_layers = getattr(parent, attr)
        n = len(all_layers)
        to_drop = max(1, int(n * 50 / 100))
        kept = n - to_drop
        model._unloaded_layers = all_layers[kept:]
        setattr(parent, attr, all_layers[:kept])

        self.assertEqual(len(getattr(parent, attr)), kept)

        unloaded = getattr(model, '_unloaded_layers', None)
        if unloaded is not None:
            current = getattr(parent, attr)
            setattr(parent, attr, list(current) + list(unloaded))
            del model._unloaded_layers

        self.assertEqual(len(getattr(parent, attr)), n)
        self.assertFalse(hasattr(model, '_unloaded_layers'))

    def test_unload_clears_cache_refs(self):
        model = FakeModel(n_layers=6)
        parent, attr = model, "layers"
        all_layers = getattr(parent, attr)
        to_drop = 3
        kept = 3
        removed = all_layers[kept:]
        model._unloaded_layers = removed
        setattr(parent, attr, all_layers[:kept])

        removed_refs = [id(w) for w in removed]
        active_ids = [id(w) for w in getattr(parent, attr)]
        for rid in removed_refs:
            self.assertNotIn(rid, active_ids)


if __name__ == "__main__":
    unittest.main()
