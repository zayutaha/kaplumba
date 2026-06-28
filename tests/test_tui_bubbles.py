import unittest


class TestCopyableMarkdown(unittest.TestCase):

    def test_copyable_markdown_has_no_on_click(self):
        """CopyableMarkdown should not override on_click (relies on Textual selection)."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.CopyableMarkdown)
        self.assertNotIn("on_click", src)

    def test_chatui_has_no_custom_copy_handler(self):
        """ChatUI should not define custom copy handlers (uses Textual built-in)."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.ChatUI)
        self.assertNotIn("_copy_single", src,
                         "Should use Textual's built-in text selection, not custom copy")
        self.assertNotIn("on_click", src,
                         "Should use Textual's built-in text selection, not click handler")


if __name__ == "__main__":
    unittest.main()
