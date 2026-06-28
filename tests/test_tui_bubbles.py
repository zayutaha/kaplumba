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
        # Should walk parent chain (not just check event.widget directly)
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


if __name__ == "__main__":
    unittest.main()
