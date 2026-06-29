import asyncio
import unittest
from unittest.mock import patch


class TestCopyableMarkdown(unittest.TestCase):

    def test_copyable_markdown_has_no_on_click(self):
        """CopyableMarkdown should not override on_click (relies on ChatUI.on_click)."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.CopyableMarkdown)
        self.assertNotIn("on_click", src)

    def test_copy_single_calls_pbcopy(self):
        """_copy_single should pipe text to pbcopy."""
        import tui_main
        async def run():
            await tui_main._copy_single("Hello world")

        with patch.object(asyncio, "create_subprocess_exec") as mock_pbcopy:
            asyncio.run(run())

        mock_pbcopy.assert_called_once()

    def test_copy_single_empty_does_nothing(self):
        """_copy_single with empty string should not call pbcopy."""
        import tui_main
        async def run():
            await tui_main._copy_single("")

        with patch.object(asyncio, "create_subprocess_exec") as mock_pbcopy:
            asyncio.run(run())

        mock_pbcopy.assert_not_called()

    def test_chatui_on_click_walks_widget_tree(self):
        """ChatUI.on_click should walk up to find CopyableMarkdown ancestor."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.ChatUI.on_click)
        self.assertIn("parent", src,
                      "on_click should walk up widget tree for CopyableMarkdown")

    def test_chatui_on_click_notifies(self):
        """ChatUI.on_click should call notify when copying."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.ChatUI.on_click)
        self.assertIn("notify", src,
                      "on_click should notify user on copy")

    def test_chatui_on_click_uses_raw_text(self):
        """ChatUI.on_click should prefer _raw_text over _markdown to avoid cursor/stopped."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.ChatUI.on_click)
        self.assertIn("_raw_text", src,
                      "on_click should use _raw_text for copy, not _markdown")

    def test_chatui_on_click_handles_meta_modifier(self):
        """ChatUI.on_click should check event.meta for raw/debug copy."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.ChatUI.on_click)
        self.assertIn("event.meta", src,
                      "on_click should use event.meta for debug copy")
        self.assertIn("Raw copied", src,
                      "on_click should show 'Raw copied' on meta+click")


class TestOptionsLiveUpdate(unittest.IsolatedAsyncioTestCase):

    def test_handle_options_hides_container(self):
        """handle_options_selected should hide #options-selector-container."""
        import orchestrator
        import inspect
        src = inspect.getsource(orchestrator.Orchestrator.handle_options_selected)
        self.assertIn("options-selector-container", src,
                      "should hide options container after applying")

    def test_handle_options_sends_set_option_for_runtime_keys(self):
        """handle_options_selected should send /set_option for temp/top_p/etc."""
        import orchestrator
        import inspect
        src = inspect.getsource(orchestrator.Orchestrator.handle_options_selected)
        self.assertIn("/set_option", src,
                      "should send /set_option commands to subprocess")

    def test_handle_options_reloads_for_turbo_keys(self):
        """handle_options_selected should reload for turbo_kv_bits changes."""
        import orchestrator
        import inspect
        src = inspect.getsource(orchestrator.Orchestrator.handle_options_selected)
        self.assertIn("turbo_kv_bits", src,
                      "should check for turbo_kv_bits changes")
        self.assertIn("_load_model", src,
                      "should reload model for restart-required options")

    async def test_options_selector_closes_on_enter_with_running_model(self):
        """Pressing Enter in options with a running model should close the selector."""
        from tui_main import ChatUI
        from typing import AsyncIterator

        class StubPortOptions:
            def __init__(self):
                self.running = True
                self.commands = []

            async def start(self, model_path, options, system_prompt):
                return True

            async def send_message(self, text: str) -> AsyncIterator[str]:
                return
                yield

            async def send_kaplumbebek_message(self, text: str) -> AsyncIterator[str]:
                return
                yield

            async def send_command(self, text: str, timeout: int = 60) -> str | None:
                self.commands.append(text)
                return None

            async def interrupt(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        port = StubPortOptions()
        app = ChatUI(port=port)
        async with app.run_test() as pilot:
            app.controller.selected_model = "test-model"
            app.show_chat_ui()
            await pilot.pause()

            # Open options
            app.query_one("#input").text = "/options"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            # Options container should be visible
            container = app.query_one("#options-selector-container")
            self.assertTrue(container.display,
                            "Options selector should be visible after /options")

            # Change temp (index 0): increase value
            sel = app.query_one("#options-selector")
            sel.selected_index = 0  # first row = temp
            sel.change_value(1)     # increase temp
            await pilot.pause()

            # Press Enter to apply
            await pilot.press("enter")
            await pilot.pause()

            # Container should be hidden
            self.assertFalse(container.display,
                             "Options selector should close after applying")

            # Should have sent /set_option commands
            self.assertGreater(len(port.commands), 0,
                               "Should send /set_option commands")
            self.assertTrue(
                any("/set_option temp=" in c for c in port.commands),
                "Should include /set_option for temp")


if __name__ == "__main__":
    unittest.main()
