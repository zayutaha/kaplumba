import asyncio
import unittest
from typing import AsyncIterator

from tui_main import ChatUI


class StubPort:
    def __init__(self):
        self.running = True

    async def start(self, model_path, options, system_prompt):
        return True

    async def send_message(self, text: str) -> AsyncIterator[str]:
        return
        yield

    async def send_minichat_message(self, text: str) -> AsyncIterator[str]:
        for chunk in ["Hello", " from", " mini", " chat"]:
            yield chunk

    async def send_command(self, text: str, timeout: int = 60) -> str | None:
        return None

    async def interrupt(self) -> None:
        pass

    async def stop(self) -> None:
        self.running = False


class TestMiniChat(unittest.IsolatedAsyncioTestCase):

    async def test_ctrl_o_opens_minichat_screen(self):
        """Ctrl+O should push MiniChatScreen when model is idle."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            self.assertFalse(
                any(isinstance(s, type(app.query_one("#chat-center"))) for s in app.screen_stack)
            )

            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.minichat_popup import MiniChatScreen
            self.assertTrue(
                any(isinstance(s, MiniChatScreen) for s in app.screen_stack),
                "MiniChatScreen should be pushed after Ctrl+O",
            )

    async def test_ctrl_o_closes_minichat_screen(self):
        """Ctrl+O should pop MiniChatScreen when already open."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.minichat_popup import MiniChatScreen
            self.assertTrue(
                any(isinstance(s, MiniChatScreen) for s in app.screen_stack)
            )

            await pilot.press("ctrl+o")
            await pilot.pause()

            self.assertFalse(
                any(isinstance(s, MiniChatScreen) for s in app.screen_stack),
                "MiniChatScreen should be popped after second Ctrl+O",
            )

    async def test_escape_closes_minichat_screen(self):
        """Escape should close the MiniChatScreen."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            from textual_ui.widgets.minichat_popup import MiniChatScreen
            self.assertFalse(
                any(isinstance(s, MiniChatScreen) for s in app.screen_stack),
                "MiniChatScreen should be popped after Escape",
            )

    async def test_minichat_not_opened_when_busy(self):
        """Ctrl+O should do nothing when model is busy generating."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            app.busy = True  # set after on_mount resets it
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.minichat_popup import MiniChatScreen
            self.assertFalse(
                any(isinstance(s, MiniChatScreen) for s in app.screen_stack),
                "MiniChatScreen should not open when busy",
            )

    async def test_minichat_sends_message_and_shows_bubbles(self):
        """Sending a message in mini-chat should show user + assistant bubbles."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.minichat_popup import MiniChatScreen
            screen = next(s for s in app.screen_stack if isinstance(s, MiniChatScreen))

            # Type and send a message
            input_widget = screen.query_one("#minichat-input")
            input_widget.text = "Hello!"
            await pilot.pause()

            # Press Enter to send
            await pilot.press("enter")
            await pilot.pause()

            # Check that bubbles appeared
            chat_area = screen.query_one("#minichat-chat")
            children = list(chat_area.children)
            user_bubbles = [c for c in children if "mc-user" in c.classes]
            assistant_bubbles = [c for c in children if "mc-assistant" in c.classes]
            self.assertEqual(len(user_bubbles), 1)
            self.assertGreaterEqual(len(assistant_bubbles), 1)

            # Check history was updated
            self.assertEqual(len(app.controller.minichat_history), 2)

    async def test_minichat_history_persists_across_close_reopen(self):
        """Mini-chat history should survive closing and re-opening the popup."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            # Send a message
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.minichat_popup import MiniChatScreen
            screen = next(s for s in app.screen_stack if isinstance(s, MiniChatScreen))
            input_widget = screen.query_one("#minichat-input")
            input_widget.text = "Persist me"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            # Close
            await pilot.press("escape")
            await pilot.pause()

            # Re-open
            await pilot.press("ctrl+o")
            await pilot.pause()

            screen = next(s for s in app.screen_stack if isinstance(s, MiniChatScreen))
            chat_area = screen.query_one("#minichat-chat")
            children = list(chat_area.children)
            self.assertGreaterEqual(len(children), 2,
                                    "History should be restored on re-open")


if __name__ == "__main__":
    unittest.main()
