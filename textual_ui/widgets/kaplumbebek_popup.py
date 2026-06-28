import asyncio

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click, Key
from textual.screen import Screen
from textual.widgets import Markdown, Static, TextArea


class _KaplumbebekInput(TextArea):
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
            if isinstance(screen, KaplumbebekScreen):
                await screen._send_message()
            return
        await super()._on_key(event)


class KaplumbebekScreen(Screen):
    CSS = """
    KaplumbebekScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.75);
    }
    #kb-box {
        width: 66;
        height: 75%;
        background: #1a1a1a;
        border: round #f0a500;
        layout: vertical;
    }
    #kb-title {
        height: 3;
        padding: 1 2;
        text-style: bold;
        background: #252525;
        color: #f0a500;
    }
    #kb-chat {
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
        overflow-x: hidden;
    }
    #kb-input-box {
        height: auto;
        background: #161616;
        border-top: solid #303030;
        padding: 0 1;
    }
    #kb-input {
        width: 1fr;
        margin: 0 1;
        background: #161616;
        color: #e0e0e0;
        border: none;
    }
    .kb-user {
        margin-top: 1;
        padding: 1 2;
        background: #222;
        border: round #333;
        color: #d8d8d8;
    }
    .kb-assistant {
        margin-bottom: 1;
        padding: 1 2 0 2;
        color: #f0a500;
        border: round transparent;
    }
    #kb-send-btn {
        width: 8;
        background: #f0a500;
        color: #000;
        text-style: bold;
        text-align: center;
        content-align: center middle;
        height: 100%;
    }
    #kb-send-btn.stopping {
        background: #e05a5a;
        color: #fff;
    }
    .kb-flash {
        background: #3a3a1a !important;
        border: round #f0a500 !important;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="kb-box"):
            yield Static(" Kaplumbebek  [dim](Esc to close)[/]", id="kb-title")
            yield VerticalScroll(id="kb-chat")
            with Horizontal(id="kb-input-box"):
                yield _KaplumbebekInput("", id="kb-input")
                yield Static(" SEND ", id="kb-send-btn")

    async def on_mount(self):
        self._chat = self.query_one("#kb-chat", VerticalScroll)
        self._input = self.query_one("#kb-input", _KaplumbebekInput)
        self._send_btn = self.query_one("#kb-send-btn", Static)
        self._streaming = False
        await self._reload_history()
        self._input.focus()

    @on(Click, "#kb-send-btn")
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
        for msg in self.app.controller.kaplumbebek_history:
            cls = "kb-user" if msg["role"] == "user" else "kb-assistant"
            await self._chat.mount(Markdown(msg["content"], classes=cls))
        self._chat.scroll_end(animate=False)

    async def _add_bubble(self, text: str, role: str):
        cls = "kb-user" if role == "user" else "kb-assistant"
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
        self.app.controller.kaplumbebek_history.append({"role": "user", "content": text})
        self._streaming = True
        self._update_send_btn()

        assistant = await self._add_bubble("", "assistant")
        full = ""
        try:
            async for chunk in self.app.controller.send_kaplumbebek(text):
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
                self.app.controller.kaplumbebek_history.append({"role": "assistant", "content": display})
        elif full:
            assistant.update(full)
            self.app.controller.kaplumbebek_history.append({"role": "assistant", "content": full})
        else:
            assistant.update("*no response*")

        self._streaming = False
        self._update_send_btn()
        self._input.focus()

    async def on_screen_resume(self):
        await self._reload_history()
        self._input.focus()

    async def on_click(self, event: Click):
        if not event.ctrl:
            return
        widget = event.widget
        while widget is not None:
            if isinstance(widget, Markdown):
                text = getattr(widget, "_markdown", "") or ""
                if event.alt:
                    text = getattr(widget, "_raw_text", "") or text
                    label = "Raw copied"
                else:
                    label = "Copied"
                if text:
                    await self._copy_text(text)
                    widget.add_class("kb-flash")
                    self.set_timer(0.2, lambda w=widget: w.remove_class("kb-flash"))
                    self.app.notify(label, timeout=2)
                event.stop()
                return
            widget = widget.parent

    async def _copy_text(self, text: str):
        if not text:
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                "pbcopy", stdin=asyncio.subprocess.PIPE,
            )
            await proc.communicate(input=text.encode())
        except FileNotFoundError:
            pass
