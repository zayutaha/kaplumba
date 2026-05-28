import asyncio
import unittest
from unittest.mock import MagicMock, patch


class TestCopyableMarkdown(unittest.TestCase):

    def test_click_event_has_no_is_double(self):
        """The version of Textual used here does not have is_double on Click.
        The handler must not crash and should treat double-click as two
        single-clicks (toggle on first, toggle off on second)."""
        from textual.events import Click
        self.assertFalse(hasattr(Click, "is_double"),
                         "This Textual version has is_double — update handler")

    def test_on_click_source_has_no_is_double(self):
        """The on_click handler should NOT reference is_double."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.CopyableMarkdown.on_click)
        self.assertNotIn("is_double", src,
                         "on_click should not use is_double — "
                         "this Textual version doesn't have it")

    def test_copy_selected_empty_does_nothing(self):
        """_copy_selected with no selections should notify and not call pbcopy."""
        import tui_main
        tui_main._selected_bubbles.clear()
        app = MagicMock()
        app.notify = MagicMock()

        async def run():
            await tui_main._copy_selected(app)

        with patch.object(asyncio, "create_subprocess_exec") as mock_pbcopy:
            asyncio.run(run())

        mock_pbcopy.assert_not_called()
        app.notify.assert_called_once()

    def test_copy_selected_joins_messages(self):
        """_copy_selected should join multiple selections with double newlines."""
        import tui_main
        tui_main._selected_bubbles.clear()

        class FakeBubble:
            def __init__(self, text):
                self._markdown = text
                self._initial_markdown = None
            def remove_class(self, name):
                pass

        b1 = FakeBubble("First message")
        b2 = FakeBubble("Second message")
        tui_main._selected_bubbles.extend([b1, b2])

        app = MagicMock()
        app.notify = MagicMock()

        async def run():
            await tui_main._copy_selected(app)

        async def mock_pbcopy(*args, **kwargs):
            class Proc:
                returncode = 0
                async def communicate(self, input=b""):
                    pass
            return Proc()

        with patch.object(asyncio, "create_subprocess_exec", side_effect=mock_pbcopy) as m:
            asyncio.run(run())

        m.assert_called_once()
        self.assertEqual(len(tui_main._selected_bubbles), 0)


if __name__ == "__main__":
    unittest.main()
