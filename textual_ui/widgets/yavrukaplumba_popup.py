import asyncio

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click, Key
from textual.screen import Screen
from textual.widgets import Markdown, Static, TextArea


class _YavruInput(TextArea):
    def on_mount(self):
        self.show_line_numbers = False
        self.soft_wrap = True
        self.styles.height = 1

    async def _on_key(self, event: Key):
        if event.key is None:
            return
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            screen = self.app.screen
            if isinstance(screen, YavrukaplumbaScreen):
                await screen._send_message()
            return
        await super()._on_key(event)


class YavrukaplumbaScreen(Screen):
    CSS = """
    YavrukaplumbaScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.75);
    }
    #yavru-box {
        width: 66;
        height: 75%;
        background: #1a1a1a;
        border: round #f0a500;
        layout: vertical;
    }
    #yavru-title {
        height: 3;
        padding: 1 2;
        text-style: bold;
        background: #252525;
        color: #f0a500;
    }
    #yavru-chat {
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
        overflow-x: hidden;
    }
    #yavru-input-box {
        height: auto;
        background: #161616;
        border-top: solid #303030;
        padding: 0 1;
    }
    #yavru-input {
        width: 1fr;
        margin: 0 1;
        background: #161616;
        color: #e0e0e0;
        border: none;
    }
    .yv-user {
        margin-top: 1;
        padding: 1 2;
        background: #222;
        border: round #333;
        color: #d8d8d8;
    }
    .yv-assistant {
        margin-bottom: 1;
        padding: 1 2 0 2;
        color: #f0a500;
        border: round transparent;
    }
    #yv-send-btn {
        width: 8;
        background: #f0a500;
        color: #000;
        text-style: bold;
        text-align: center;
        content-align: center middle;
        height: 100%;
    }
    #yv-send-btn.stopping {
        background: #e05a5a;
        color: #fff;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="yavru-box"):
            yield Static(" Yavru  [dim](Esc to close)[/]", id="yavru-title")
            yield VerticalScroll(id="yavru-chat")
            with Horizontal(id="yavru-input-box"):
                yield _YavruInput("", id="yavru-input")
                yield Static(" SEND ", id="yv-send-btn")

    async def on_mount(self):
        self._chat = self.query_one("#yavru-chat", VerticalScroll)
        self._input = self.query_one("#yavru-input", _YavruInput)
        self._send_btn = self.query_one("#yv-send-btn", Static)
        self._streaming = False
        await self._reload_history()
        self._input.focus()

    @on(Click, "#yv-send-btn")
    async def on_send_click(self):
        if self._streaming:
            self._streaming = False
            asyncio.create_task(self._interrupt_stream())
        else:
            await self._send_message()

    def _update_send_btn(self):
        if self._streaming:
            self._send_btn.update(" STOP ")
            self._send_btn.add_class("stopping")
        else:
            self._send_btn.update(" SEND ")
            self._send_btn.remove_class("stopping")

    async def _reload_history(self):
        self._chat.remove_children()
        for msg in self.app.controller.yavru_history:
            cls = "yv-user" if msg["role"] == "user" else "yv-assistant"
            await self._chat.mount(Markdown(msg["content"], classes=cls))
        self._chat.scroll_end(animate=False)

    async def _add_bubble(self, text: str, role: str):
        cls = "yv-user" if role == "user" else "yv-assistant"
        bubble = Markdown(text, classes=cls)
        await self._chat.mount(bubble)
        self._chat.scroll_end(animate=False)
        return bubble

    async def _interrupt_stream(self):
        try:
            await self.app.controller.port.interrupt()
        except Exception:
            pass
        self._update_send_btn()

    async def _send_message(self):
        if self._streaming:
            return

        text = self._input.text.strip()
        if not text:
            return
        self._input.clear()

        await self._add_bubble(text, "user")
        self.app.controller.yavru_history.append({"role": "user", "content": text})
        self._streaming = True
        self._update_send_btn()

        assistant = await self._add_bubble("", "assistant")
        full = ""
        try:
            async for chunk in self.app.controller.send_yavru(text):
                full += chunk
                assistant.update(full + " ▌")
        except Exception:
            assistant.update("*error*")
            self._streaming = False
            self._update_send_btn()
            self._input.focus()
            return

        if not self._streaming:
            suffix = "\n\n*stopped*"
            display = full + suffix if full else suffix
            assistant.update(display)
            if full:
                self.app.controller.yavru_history.append({"role": "assistant", "content": display})
        elif full:
            assistant.update(full)
            self.app.controller.yavru_history.append({"role": "assistant", "content": full})
        else:
            assistant.update("*no response*")

        self._streaming = False
        self._update_send_btn()
        self._input.focus()

    async def on_screen_resume(self):
        await self._reload_history()
        self._input.focus()
