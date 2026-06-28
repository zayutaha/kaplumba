from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.screen import Screen
from textual.widgets import Static, TextArea


class _MiniChatInput(TextArea):
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
            if isinstance(screen, MiniChatScreen):
                await screen._send_message()
            return
        await super()._on_key(event)


class MiniChatScreen(Screen):
    CSS = """
    MiniChatScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.75);
    }
    #minichat-box {
        width: 66;
        height: 75%;
        background: #1a1a1a;
        border: round #f0a500;
        layout: vertical;
    }
    #minichat-title {
        height: 3;
        padding: 1 2;
        text-style: bold;
        background: #252525;
        color: #f0a500;
    }
    #minichat-chat {
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
        overflow-x: hidden;
    }
    #minichat-input-box {
        height: auto;
        background: #161616;
        border-top: solid #303030;
        padding: 0 1;
    }
    #minichat-input {
        width: 1fr;
        margin: 0 1;
        background: #161616;
        color: #e0e0e0;
        border: none;
    }
    .mc-user {
        margin-top: 1;
        padding: 1 2;
        background: #222;
        border: round #333;
        color: #d8d8d8;
    }
    .mc-assistant {
        margin-bottom: 1;
        padding: 1 2 0 2;
        color: #f0a500;
        border: round transparent;
    }
    #minichat-status {
        height: 1;
        padding: 0 2;
        color: #666;
        text-style: italic;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="minichat-box"):
            yield Static(" Mini Chat  [dim](Esc to close)[/]", id="minichat-title")
            yield VerticalScroll(id="minichat-chat")
            yield Static("", id="minichat-status")
            with Horizontal(id="minichat-input-box"):
                yield _MiniChatInput("", id="minichat-input")

    async def on_mount(self):
        self._chat = self.query_one("#minichat-chat", VerticalScroll)
        self._input = self.query_one("#minichat-input", _MiniChatInput)
        self._status = self.query_one("#minichat-status", Static)
        self._streaming = False
        await self._reload_history()
        self._input.focus()

    async def _reload_history(self):
        self._chat.remove_children()
        for msg in self.app.controller.minichat_history:
            cls = "mc-user" if msg["role"] == "user" else "mc-assistant"
            await self._chat.mount(Static(msg["content"], classes=cls))
        self._chat.scroll_end(animate=False)

    async def _add_bubble(self, text: str, role: str):
        cls = "mc-user" if role == "user" else "mc-assistant"
        bubble = Static(text, classes=cls)
        await self._chat.mount(bubble)
        self._chat.scroll_end(animate=False)
        return bubble

    def on_key(self, event: Key):
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.app.pop_screen()

    async def _send_message(self):
        if self._streaming:
            return

        text = self._input.text.strip()
        if not text:
            return
        self._input.clear()

        await self._add_bubble(text, "user")
        self.app.controller.minichat_history.append({"role": "user", "content": text})
        self._streaming = True
        self._status.update("Thinking...")

        assistant = await self._add_bubble("", "assistant")
        full = ""
        try:
            async for chunk in self.app.controller.send_minichat(text):
                full += chunk
                assistant.update(full + " ▌")
        except Exception:
            assistant.update("*error*")
            self._streaming = False
            self._status.update("")
            self._input.focus()
            return

        if full:
            assistant.update(full)
            self.app.controller.minichat_history.append({"role": "assistant", "content": full})
        else:
            assistant.update("*no response*")

        self._streaming = False
        self._status.update("")
        self._input.focus()

    async def on_screen_resume(self):
        await self._reload_history()
        self._input.focus()
