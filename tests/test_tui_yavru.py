import asyncio
import unittest
from typing import AsyncIterator

from textual.widgets import Static

from tui_main import ChatUI


class StubPort:
    def __init__(self):
        self.running = True
        self._interrupted = False

    async def start(self, model_path, options, system_prompt):
        return True

    async def send_message(self, text: str) -> AsyncIterator[str]:
        return
        yield

    async def send_yavru_message(self, text: str) -> AsyncIterator[str]:
        for chunk in ["Hello", " from", " yavru", " chat"]:
            if self._interrupted:
                break
            await asyncio.sleep(0.03)
            yield chunk

    async def send_command(self, text: str, timeout: int = 60) -> str | None:
        return None

    async def interrupt(self) -> None:
        self._interrupted = True

    async def stop(self) -> None:
        self.running = False


class TestYavru(unittest.IsolatedAsyncioTestCase):

    async def test_ctrl_o_opens_yavru_screen(self):
        """Ctrl+O should push YavrukaplumbaScreen when model is idle."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            self.assertFalse(
                any(isinstance(s, type(app.query_one("#chat-center"))) for s in app.screen_stack)
            )

            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.yavrukaplumba_popup import YavrukaplumbaScreen
            self.assertTrue(
                any(isinstance(s, YavrukaplumbaScreen) for s in app.screen_stack),
                "YavrukaplumbaScreen should be pushed after Ctrl+O",
            )

    async def test_ctrl_o_closes_yavru_screen(self):
        """Ctrl+O should pop YavrukaplumbaScreen when already open."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.yavrukaplumba_popup import YavrukaplumbaScreen
            self.assertTrue(
                any(isinstance(s, YavrukaplumbaScreen) for s in app.screen_stack)
            )

            await pilot.press("ctrl+o")
            await pilot.pause()

            self.assertFalse(
                any(isinstance(s, YavrukaplumbaScreen) for s in app.screen_stack),
                "YavrukaplumbaScreen should be popped after second Ctrl+O",
            )

    async def test_escape_does_nothing_on_yavru(self):
        """Escape should have no effect on YavrukaplumbaScreen."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.yavrukaplumba_popup import YavrukaplumbaScreen
            await pilot.press("escape")
            await pilot.pause()

            self.assertTrue(
                any(isinstance(s, YavrukaplumbaScreen) for s in app.screen_stack),
                "Escape should not close Yavru screen",
            )

    async def test_yavru_not_opened_when_busy(self):
        """Ctrl+O should do nothing when model is busy generating."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            app.busy = True
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.yavrukaplumba_popup import YavrukaplumbaScreen
            self.assertFalse(
                any(isinstance(s, YavrukaplumbaScreen) for s in app.screen_stack),
                "YavrukaplumbaScreen should not open when busy",
            )

    async def test_yavru_sends_message_and_shows_bubbles(self):
        """Sending a message in yavru should show user + assistant bubbles."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.yavrukaplumba_popup import YavrukaplumbaScreen
            screen = next(s for s in app.screen_stack if isinstance(s, YavrukaplumbaScreen))

            input_widget = screen.query_one("#yavru-input")
            input_widget.text = "Hello!"
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            chat_area = screen.query_one("#yavru-chat")
            children = list(chat_area.children)
            user_bubbles = [c for c in children if "yv-user" in c.classes]
            assistant_bubbles = [c for c in children if "yv-assistant" in c.classes]
            self.assertEqual(len(user_bubbles), 1)
            self.assertGreaterEqual(len(assistant_bubbles), 1)

            self.assertEqual(len(app.controller.yavru_history), 2)

    async def test_yavru_history_persists_across_close_reopen(self):
        """Yavru history should survive closing and re-opening the popup."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.yavrukaplumba_popup import YavrukaplumbaScreen
            screen = next(s for s in app.screen_stack if isinstance(s, YavrukaplumbaScreen))
            input_widget = screen.query_one("#yavru-input")
            input_widget.text = "Persist me"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            # Close via Ctrl+O (Escape does nothing on yavru)
            await pilot.press("ctrl+o")
            await pilot.pause()

            await pilot.press("ctrl+o")
            await pilot.pause()

            screen = next(s for s in app.screen_stack if isinstance(s, YavrukaplumbaScreen))
            chat_area = screen.query_one("#yavru-chat")
            children = list(chat_area.children)
            self.assertGreaterEqual(len(children), 2,
                                    "History should be restored on re-open")

    async def test_escape_does_nothing_during_stream(self):
        """Escape should not interrupt or close yavru during streaming."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.yavrukaplumba_popup import YavrukaplumbaScreen
            screen = next(s for s in app.screen_stack if isinstance(s, YavrukaplumbaScreen))
            screen._streaming = True
            await pilot.press("escape")
            await pilot.pause()

            port = app.controller.port
            self.assertFalse(port._interrupted,
                             "Escape should not interrupt during stream")
            self.assertTrue(screen._streaming,
                            "_streaming should remain True after Escape")
            self.assertTrue(
                any(isinstance(s, YavrukaplumbaScreen) for s in app.screen_stack),
                "Escape should not close yavru during stream",
            )

    async def test_send_button_toggles_to_stop_during_stream(self):
        """Send button should show STOP when streaming, SEND when idle."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.yavrukaplumba_popup import YavrukaplumbaScreen
            screen = next(s for s in app.screen_stack if isinstance(s, YavrukaplumbaScreen))
            btn = screen.query_one("#yv-send-btn", Static)
            self.assertIn("SEND", str(btn.render()))

            screen._streaming = True
            screen._update_send_btn()
            self.assertIn("STOP", str(btn.render()))
            self.assertTrue(btn.has_class("stopping"))

            screen._streaming = False
            screen._update_send_btn()
            self.assertIn("SEND", str(btn.render()))
            self.assertFalse(btn.has_class("stopping"))

    async def test_send_button_click_interrupts_stream(self):
        """Clicking STOP button during stream should interrupt."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()

            from textual_ui.widgets.yavrukaplumba_popup import YavrukaplumbaScreen
            screen = next(s for s in app.screen_stack if isinstance(s, YavrukaplumbaScreen))
            screen._streaming = True
            screen._update_send_btn()

            await pilot.click("#yv-send-btn")
            await pilot.pause()

            port = app.controller.port
            self.assertTrue(port._interrupted,
                            "Clicking STOP should interrupt the port")

    async def test_yavru_blocked_in_main_chat(self):
        """Orchestrator.handle_submit should reject /yavru before sending to model."""
        import inspect
        import orchestrator
        src = inspect.getsource(orchestrator.Orchestrator.handle_submit)
        self.assertIn("/yavru", src,
                       "handle_submit should check for /yavru")
        # It should return early, not fall through to send_message
        send_idx = src.find("send_message")
        yavru_idx = src.find("/yavru")
        self.assertLess(yavru_idx, send_idx if send_idx > 0 else len(src),
                        "/yavru check should come before send_message")
