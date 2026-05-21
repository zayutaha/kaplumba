import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_catalog import ModelCapabilities, ModelInfo
from settings_store import DEFAULT_MODEL_OPTIONS
from textual_ui.widgets.options_selector import OptionsSelector
from tui_main import ChatUI


class StubPort:
    def __init__(self, running: bool = False, chunks: list[str] | None = None):
        self.running = running
        self.chunks = chunks or ["Hello", " world", "!"]
        self.started: list[tuple[str, dict, str]] = []
        self.messages: list[str] = []
        self.commands: list[str] = []
        self.interrupts = 0
        self.stops = 0

    async def start(self, model_path: str, options: dict, system_prompt: str) -> bool:
        self.started.append((model_path, dict(options), system_prompt))
        self.running = True
        return True

    async def send_message(self, text: str):
        self.messages.append(text)
        for chunk in self.chunks:
            yield chunk

    async def send_command(self, text: str, timeout: int = 60) -> None:
        self.commands.append(text)

    async def interrupt(self) -> None:
        self.interrupts += 1

    async def stop(self) -> None:
        self.stops += 1
        self.running = False


def sample_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            name="demo-model",
            size_bytes=1024,
            size_gib="0.0 GB",
            capabilities=ModelCapabilities(),
        )
    ]


class TestChatUINavigation(unittest.IsolatedAsyncioTestCase):
    async def test_escape_from_options_returns_to_model_picker_when_not_running(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=False))
            async with app.run_test() as pilot:
                await app.show_options_selector()
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

                self.assertTrue(app.query_one("#model-selector-container").display)
                self.assertFalse(app.query_one("#options-selector-container").display)

    async def test_escape_from_options_returns_to_chat_when_running(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=True))
            async with app.run_test() as pilot:
                app.show_chat_ui()
                await app.show_options_selector()
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

                self.assertTrue(app.query_one("#chat-center").display)
                self.assertTrue(app.query_one("#input-center").display)
                self.assertFalse(app.query_one("#options-selector-container").display)

    async def test_escape_from_model_picker_returns_to_chat_when_running(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=True))
            async with app.run_test() as pilot:
                app.show_chat_ui()
                await app.show_model_selector()
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

                self.assertTrue(app.query_one("#chat-center").display)
                self.assertTrue(app.query_one("#input-center").display)
                self.assertFalse(app.query_one("#model-selector-container").display)

    async def test_show_options_selector_uses_controller_state(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=False))
            app.controller.model_options = {**DEFAULT_MODEL_OPTIONS, "temp": 0.2, "mtp": False}
            async with app.run_test() as pilot:
                await app.show_options_selector()
                await pilot.pause()
                selector = app.query_one("#options-selector")
                self.assertEqual(selector.options["temp"], 0.2)
                self.assertFalse(selector.options["mtp"])

    async def test_model_editor_loads_saved_config_and_escape_persists_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_configs.json"
            config_path.write_text(json.dumps({
                "demo-model": {
                    "options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.3, "mtp": False},
                    "personality": "historian",
                }
            }))
            with patch("tui_main.list_models", return_value=sample_models()):
                with patch("settings_store.MODEL_CONFIGS_PATH", config_path):
                    app = ChatUI(port=StubPort(running=False))
                    async with app.run_test() as pilot:
                        await app.action_model_edit("demo-model")
                        await pilot.pause()

                        sel = app.query_one("#options-selector", OptionsSelector)
                        self.assertEqual(sel.model_name, "demo-model")
                        self.assertEqual(sel.options["temp"], 0.3)
                        self.assertEqual(sel.personality, "historian")

                        await pilot.press("escape")
                        await pilot.pause()

                        saved = json.loads(config_path.read_text())
                        self.assertEqual(saved["demo-model"]["options"]["temp"], 0.3)
                        self.assertEqual(saved["demo-model"]["personality"], "historian")
                        self.assertTrue(app.query_one("#model-selector-container").display)

    async def test_slash_models_opens_picker_and_escape_returns_to_chat(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=True))
            async with app.run_test() as pilot:
                app.show_chat_ui()
                app.query_one("#input").load_text("/models")
                await app.action_submit()
                await pilot.pause()

                self.assertTrue(app.query_one("#model-selector-container").display)
                await pilot.press("escape")
                await pilot.pause()
                self.assertTrue(app.query_one("#chat-center").display)

    async def test_slash_options_opens_options_and_escape_returns_to_chat(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=True))
            async with app.run_test() as pilot:
                app.show_chat_ui()
                app.query_one("#input").load_text("/options")
                await app.action_submit()
                await pilot.pause()

                self.assertTrue(app.query_one("#options-selector-container").display)
                await pilot.press("escape")
                await pilot.pause()
                self.assertTrue(app.query_one("#chat-center").display)

    async def test_slash_personality_opens_personality_and_escape_returns_to_chat(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=True))
            async with app.run_test() as pilot:
                app.show_chat_ui()
                app.query_one("#input").load_text("/personality")
                await app.action_submit()
                await pilot.pause()

                self.assertTrue(app.query_one("#personality-selector-container").display)
                await pilot.press("escape")
                await pilot.pause()
                self.assertTrue(app.query_one("#chat-center").display)

    async def test_slash_clear_resets_chat_and_sends_command(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            port = StubPort(running=True)
            app = ChatUI(port=port)
            async with app.run_test() as pilot:
                app.show_chat_ui()
                await app.handle_stream_text("hello")
                app.query_one("#input").load_text("/clear")
                await app.action_submit()
                await pilot.pause()

                child_ids = {child.id for child in app.query_one("#chat").children}
                self.assertIn("/clear", port.commands)
                self.assertIn("welcome-logo", child_ids)
                self.assertIn("welcome-prompt", child_ids)

    async def test_model_selection_transitions_from_picker_to_chat(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            port = StubPort(running=False)
            app = ChatUI(port=port)
            async with app.run_test() as pilot:
                await app.action_model_selected("demo-model")
                await pilot.pause()

                self.assertTrue(port.running)
                self.assertTrue(app.query_one("#chat-center").display)
                self.assertTrue(app.query_one("#input-center").display)
                self.assertFalse(app.query_one("#model-selector-container").display)

    async def test_generation_streams_and_can_return_to_picker_and_back(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            port = StubPort(running=True, chunks=["Hel", "lo", "!"])
            app = ChatUI(port=port)
            app.first_message = False
            async with app.run_test() as pilot:
                app.show_chat_ui()
                app.query_one("#input").load_text("hello")
                await app.action_submit()
                await app.controller._stream_task
                await pilot.pause()

                markdowns = list(app.query("#chat Markdown"))
                self.assertGreaterEqual(len(markdowns), 2)
                self.assertEqual(port.messages, ["hello"])

                app.query_one("#input").load_text("/models")
                await app.action_submit()
                await pilot.pause()
                self.assertTrue(app.query_one("#model-selector-container").display)

                await pilot.press("escape")
                await pilot.pause()
                self.assertTrue(app.query_one("#chat-center").display)

    async def test_options_selector_edit_enter_and_exit(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=False))
            async with app.run_test() as pilot:
                await app.show_options_selector()
                await pilot.pause()
                sel = app.query_one("#options-selector")
                sel.selected_index = 0  # temp

                # press e to enter edit mode
                await pilot.press("e")
                await pilot.pause()
                self.assertTrue(sel._editing)
                self.assertEqual(sel._edit_buffer, "")

                # type a value
                for ch in "0.5":
                    await pilot.press(ch)
                await pilot.pause()
                self.assertEqual(sel._edit_buffer, "0.5")

                # commit
                await pilot.press("enter")
                await pilot.pause()
                self.assertFalse(sel._editing)
                self.assertEqual(sel.options["temp"], 0.5)

    async def test_options_selector_edit_escape_cancels(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=False))
            app.controller.model_options = {**DEFAULT_MODEL_OPTIONS, "temp": 0.7}
            async with app.run_test() as pilot:
                await app.show_options_selector()
                await pilot.pause()
                sel = app.query_one("#options-selector")
                sel.selected_index = 0

                await pilot.press("e")
                for ch in "0.1":
                    await pilot.press(ch)
                await pilot.press("escape")
                await pilot.pause()
                self.assertFalse(sel._editing)
                self.assertEqual(sel.options["temp"], 0.7)

    async def test_options_selector_edit_backspace(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=False))
            async with app.run_test() as pilot:
                await app.show_options_selector()
                await pilot.pause()
                sel = app.query_one("#options-selector")
                sel.selected_index = 0

                await pilot.press("e")
                for ch in "123":
                    await pilot.press(ch)
                await pilot.pause()
                self.assertEqual(sel._edit_buffer, "123")

                await pilot.press("backspace")
                await pilot.pause()
                self.assertEqual(sel._edit_buffer, "12")

    async def test_options_selector_edit_preserves_int_value(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=False))
            async with app.run_test() as pilot:
                await app.show_options_selector()
                await pilot.pause()
                sel = app.query_one("#options-selector")
                sel.selected_index = 2  # top_k

                await pilot.press("e")
                for ch in "150":
                    await pilot.press(ch)
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(sel.options["top_k"], 150)
                self.assertIsInstance(sel.options["top_k"], int)

    async def test_options_selector_edit_preserves_float_value(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=False))
            async with app.run_test() as pilot:
                await app.show_options_selector()
                await pilot.pause()
                sel = app.query_one("#options-selector")
                sel.selected_index = 0  # temp

                await pilot.press("e")
                for ch in "0.75":
                    await pilot.press(ch)
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(sel.options["temp"], 0.75)
                self.assertIsInstance(sel.options["temp"], float)

    async def test_options_selector_edit_auto_returns_none(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=False))
            async with app.run_test() as pilot:
                await app.show_options_selector()
                await pilot.pause()
                sel = app.query_one("#options-selector")
                sel.selected_index = 6  # max_kv_size, currently None → "Auto"

                await pilot.press("e")
                for ch in "auto":
                    await pilot.press(ch)
                await pilot.press("enter")
                await pilot.pause()
                self.assertIsNone(sel.options["max_kv_size"])

    async def test_options_selector_edit_decimal_dot_supported(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=False))
            async with app.run_test() as pilot:
                await app.show_options_selector()
                await pilot.pause()
                sel = app.query_one("#options-selector")
                sel.selected_index = 8  # turbo_kv_bits

                await pilot.press("e")
                for ch in "2.5":
                    await pilot.press(ch)
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(sel.options["turbo_kv_bits"], 2.5)

    async def test_options_selector_edit_keeps_float_even_when_whole(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            app = ChatUI(port=StubPort(running=False))
            async with app.run_test() as pilot:
                await app.show_options_selector()
                await pilot.pause()
                sel = app.query_one("#options-selector")
                sel.selected_index = 0  # temp

                await pilot.press("e")
                for ch in "1.0":
                    await pilot.press(ch)
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(sel.options["temp"], 1.0)
                self.assertIsInstance(sel.options["temp"], float)

    async def test_options_selector_enter_applies_options(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            port = StubPort(running=True)
            app = ChatUI(port=port)
            app.controller.selected_model = "demo-model"
            async with app.run_test() as pilot:
                app.show_chat_ui()
                await app.show_options_selector()
                await pilot.pause()
                sel = app.query_one("#options-selector")
                sel.options["temp"] = 0.5

                await pilot.press("enter")
                await pilot.pause()

                self.assertTrue(port.running)
                self.assertEqual(app.controller.model_options["temp"], 0.5)
                self.assertTrue(app.query_one("#chat-center").display)
                self.assertFalse(app.query_one("#options-selector-container").display)

    async def test_escape_from_options_returns_to_chat_even_after_editor_mode(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            port = StubPort(running=True)
            app = ChatUI(port=port)
            app.controller.selected_model = "demo-model"
            async with app.run_test() as pilot:
                app.show_chat_ui()

                # First open model editor from picker (sets _is_editor=True)
                await app.action_model_edit("demo-model")
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

                # Now open /options and press Esc — should return to chat
                await app.show_options_selector()
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

                self.assertTrue(app.controller.port.running,
                    "Model should still be running after returning from /options")
                self.assertTrue(app.query_one("#chat-center").display,
                    "Should return to chat when model is running")
                self.assertFalse(app.query_one("#options-selector-container").display,
                    "Options selector should be hidden")
                self.assertFalse(app.query_one("#model-selector-container").display,
                    "Should NOT be in model picker")

    async def test_model_selection_loads_per_model_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_configs.json"
            config_path.write_text(json.dumps({
                "demo-model": {"options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.2}},
                "other-model": {"options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.9}},
            }))
            with patch("tui_main.list_models", return_value=sample_models() * 2):
                with patch("settings_store.MODEL_CONFIGS_PATH", config_path):
                    app = ChatUI(port=StubPort(running=False))
                    async with app.run_test() as pilot:
                        await app.action_model_selected("demo-model")
                        await pilot.pause()
                        self.assertEqual(app.controller.model_options["temp"], 0.2)

                    app2 = ChatUI(port=StubPort(running=False))
                    async with app2.run_test() as pilot2:
                        await app2.action_model_selected("other-model")
                        await pilot2.pause()
                        self.assertEqual(app2.controller.model_options["temp"], 0.9)

    async def test_options_saves_per_model_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_configs.json"
            config_path.write_text(json.dumps({
                "demo-model": {"options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.7}, "personality": "default"},
            }))
            with patch("tui_main.list_models", return_value=sample_models()):
                with patch("settings_store.MODEL_CONFIGS_PATH", config_path):
                    port = StubPort(running=True)
                    app = ChatUI(port=port)
                    app.controller.selected_model = "demo-model"
                    async with app.run_test() as pilot:
                        app.show_chat_ui()
                        # Open options, change temperature, apply
                        await app.show_options_selector()
                        await pilot.pause()
                        sel = app.query_one("#options-selector", OptionsSelector)
                        sel.options["temp"] = 0.15
                        await pilot.press("enter")
                        await pilot.pause()

                        # Check saved per-model
                        saved = json.loads(config_path.read_text())
                        self.assertEqual(saved["demo-model"]["options"]["temp"], 0.15)
                        # Model was reloaded with new options
                        self.assertEqual(app.controller.model_options["temp"], 0.15)
                        self.assertEqual(port.started[-1][1]["temp"], 0.15)

    async def test_different_models_keep_different_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_configs.json"
            config_path.write_text(json.dumps({
                "model-a": {"options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.1, "top_k": 40}, "personality": "doctor"},
                "model-b": {"options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.9, "top_k": 0}, "personality": "historian"},
            }))
            with patch("tui_main.list_models", return_value=sample_models() * 2):
                with patch("settings_store.MODEL_CONFIGS_PATH", config_path):
                    port = StubPort(running=False)
                    app = ChatUI(port=port)
                    async with app.run_test() as pilot:
                        # Load model-a
                        await app.action_model_selected("model-a")
                        await pilot.pause()
                        self.assertEqual(app.controller.model_options["temp"], 0.1)
                        self.assertEqual(app.controller.model_options["top_k"], 40)
                        self.assertEqual(app.controller.selected_personality, "doctor")

                        # Switch to model-b
                        await app.action_model_selected("model-b")
                        await pilot.pause()
                        self.assertEqual(app.controller.model_options["temp"], 0.9)
                        self.assertEqual(app.controller.model_options["top_k"], 0)
                        self.assertEqual(app.controller.selected_personality, "historian")

                        # Model-a options should be unchanged
                        configs_a = app.controller.get_model_config("model-a")
                        self.assertEqual(configs_a["options"]["temp"], 0.1)

    async def test_model_edit_from_picker_saves_per_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_configs.json"
            config_path.write_text(json.dumps({
                "demo-model": {
                    "options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.5},
                    "personality": "default",
                }
            }))
            with patch("tui_main.list_models", return_value=sample_models()):
                with patch("settings_store.MODEL_CONFIGS_PATH", config_path):
                    app = ChatUI(port=StubPort(running=False))
                    async with app.run_test() as pilot:
                        # Open model editor from picker
                        await app.action_model_edit("demo-model")
                        await pilot.pause()

                        sel = app.query_one("#options-selector", OptionsSelector)
                        self.assertEqual(sel.options["temp"], 0.5)
                        self.assertEqual(sel.personality, "default")
                        self.assertTrue(sel._is_editor)

                        # Change temp and personality, save via escape
                        sel.options["temp"] = 0.88
                        sel.personality = "doctor"
                        await pilot.press("escape")
                        await pilot.pause()

                        # Verify saved per-model
                        saved = json.loads(config_path.read_text())
                        self.assertEqual(saved["demo-model"]["options"]["temp"], 0.88)
                        self.assertEqual(saved["demo-model"]["personality"], "doctor")
                        self.assertTrue(app.query_one("#model-selector-container").display)

    async def test_model_edit_from_picker_new_model_creates_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_configs.json"
            config_path.write_text("{}")
            with patch("tui_main.list_models", return_value=sample_models()):
                with patch("settings_store.MODEL_CONFIGS_PATH", config_path):
                    app = ChatUI(port=StubPort(running=False))
                    async with app.run_test() as pilot:
                        await app.action_model_edit("new-model")
                        await pilot.pause()

                        sel = app.query_one("#options-selector", OptionsSelector)
                        # Should have defaults
                        self.assertEqual(sel.model_name, "new-model")
                        self.assertEqual(sel.personality, "default")

                        sel.options["temp"] = 0.42
                        sel.personality = "historian"
                        await pilot.press("escape")
                        await pilot.pause()

                        saved = json.loads(config_path.read_text())
                        self.assertIn("new-model", saved)
                        self.assertEqual(saved["new-model"]["options"]["temp"], 0.42)
                        self.assertEqual(saved["new-model"]["personality"], "historian")

    async def test_options_after_switching_model_shows_new_models_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_configs.json"
            config_path.write_text(json.dumps({
                "model-a": {"options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.1}},
                "model-b": {"options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.9}},
            }))
            models = [
                ModelInfo(name="model-a", size_bytes=1024, size_gib="0.0 GB", capabilities=ModelCapabilities()),
                ModelInfo(name="model-b", size_bytes=1024, size_gib="0.0 GB", capabilities=ModelCapabilities()),
            ]
            with patch("tui_main.list_models", return_value=models):
                with patch("settings_store.MODEL_CONFIGS_PATH", config_path):
                    port = StubPort(running=True)
                    app = ChatUI(port=port)
                    async with app.run_test() as pilot:
                        # Load model-a
                        await app.action_model_selected("model-a")
                        await pilot.pause()
                        await app.show_options_selector()
                        await pilot.pause()
                        sel_a = app.query_one("#options-selector", OptionsSelector)
                        self.assertEqual(sel_a.options["temp"], 0.1,
                            "Should show model-a options")

                        # Go back, switch to model-b
                        await pilot.press("escape")
                        await pilot.pause()
                        await app.action_model_selected("model-b")
                        await pilot.pause()

                        # Open options again — should show model-b options
                        await app.show_options_selector()
                        await pilot.pause()
                        sel_b = app.query_one("#options-selector", OptionsSelector)
                        self.assertEqual(sel_b.options["temp"], 0.9,
                            "Should show model-b options after switching")
                        self.assertNotEqual(sel_b.options["temp"], 0.1,
                            "Should NOT show old model-a options")

    async def test_options_after_switching_via_slash_models_shows_new_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_configs.json"
            config_path.write_text(json.dumps({
                "model-a": {"options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.1}},
                "model-b": {"options": {**DEFAULT_MODEL_OPTIONS, "temp": 0.9}},
            }))
            models = [
                ModelInfo(name="model-a", size_bytes=1024, size_gib="0.0 GB", capabilities=ModelCapabilities()),
                ModelInfo(name="model-b", size_bytes=1024, size_gib="0.0 GB", capabilities=ModelCapabilities()),
            ]
            with patch("tui_main.list_models", return_value=models):
                with patch("settings_store.MODEL_CONFIGS_PATH", config_path):
                    port = StubPort(running=True)
                    app = ChatUI(port=port)
                    async with app.run_test() as pilot:
                        # Load model-a properly via action (updates model_options)
                        await app.action_model_selected("model-a")
                        await pilot.pause()

                        # Open /options for model-a, then dismiss
                        await app.show_options_selector()
                        await pilot.pause()
                        sel = app.query_one("#options-selector")
                        self.assertEqual(sel.options["temp"], 0.1)
                        await pilot.press("escape")
                        await pilot.pause()

                        # Now switch via /models slash command
                        app.query_one("#input").load_text("/models")
                        await app.action_submit()
                        await pilot.pause()
                        # Select model-b from the picker
                        await app.action_model_selected("model-b")
                        await pilot.pause()

                        # Open /options — should now show model-b's temp=0.9
                        await app.show_options_selector()
                        await pilot.pause()
                        sel2 = app.query_one("#options-selector", OptionsSelector)
                        self.assertEqual(sel2.options["temp"], 0.9,
                            "Options should reflect model-b after switching via /models")

    async def test_ascii_kaplumba_stays_in_scrollable_chat_content_while_chatting(self):
        with patch("tui_main.list_models", return_value=sample_models()):
            port = StubPort(running=True, chunks=["Hi"])
            app = ChatUI(port=port)
            app.first_message = False
            async with app.run_test() as pilot:
                app.show_chat_ui()
                child_ids = [child.id for child in app.query_one("#chat").children]
                self.assertIn("welcome-logo", child_ids)
                self.assertIn("welcome-prompt", child_ids)

                app.query_one("#input").load_text("hello")
                await app.action_submit()
                await app.controller._stream_task
                await pilot.pause()

                child_ids = [child.id for child in app.query_one("#chat").children]
                self.assertIn("welcome-logo", child_ids)


if __name__ == "__main__":
    unittest.main()
