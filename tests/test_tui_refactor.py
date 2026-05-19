import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_catalog import ModelCapabilities, ModelInfo
from settings_store import DEFAULT_MODEL_OPTIONS
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

                        editor = app.query_one("#model-editor")
                        self.assertEqual(editor.model_name, "demo-model")
                        self.assertEqual(editor.options["temp"], 0.3)
                        self.assertEqual(editor.personality, "historian")

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
