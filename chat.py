# 🔴 MUST be first — fixes Textual crash
import textual.keys as tk

_orig = tk.key_to_character
def safe_key_to_character(key):
    if key is None:
        return None
    return _orig(key)
tk.key_to_character = safe_key_to_character


import asyncio
import re
import os
import random
from textual.app import App, ComposeResult
from textual.widgets import Markdown, TextArea, Static, Button
from textual import on
from textual.containers import VerticalScroll, Vertical, Horizontal, Center
from textual.events import Key, Click
from pylatexenc.latex2text import LatexNodes2Text
from sympy import sympify, pretty

BASE_CMD = [
    "uv", "run", "python", "-m", "mlx_lm.chat",
    "--model", "/Users/zayaantaha/.omlx/models/Qwwwen",
    "--mtp",
    "--turbo-kv-bits", "3",
    "--turbo-fp16-layers", "2",
    "--max-tokens", "16384",
]

LOGO = """
██╗  ██╗ █████╗ ██████╗ ██╗     ██╗   ██╗███╗   ███╗██████╗  █████╗ 
██║ ██╔╝██╔══██╗██╔══██╗██║     ██║   ██║████╗ ████║██╔══██╗██╔══██╗
█████╔╝ ███████║██████╔╝██║     ██║   ██║██╔████╔██║██████╔╝███████║
██╔═██╗ ██╔══██║██╔═══╝ ██║     ██║   ██║██║╚██╔╝██║██╔══██╗██╔══██║
██║  ██╗██║  ██║██║     ███████╗╚██████╔╝██║ ╚═╝ ██║██████╔╝██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝
"""

WELCOME_MESSAGES = [LOGO]

latex_converter = LatexNodes2Text()


def normalize_latex(text: str) -> str:
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    fixes = ["frac", "int", "sum", "sqrt", "sin", "cos", "tan", "log", "ln"]
    for cmd in fixes:
        text = re.sub(rf'(?<!\\)\b{cmd}\b', rf'\\{cmd}', text)
    return text


def try_sympy(expr: str) -> str:
    try:
        return f"\n```\n{pretty(sympify(expr))}\n```\n"
    except Exception:
        try:
            return latex_converter.latex_to_text(expr)
        except Exception:
            return expr


def transform_math(text: str) -> str:
    text = normalize_latex(text)
    text = re.sub(r"\$\$(.*?)\$\$", lambda m: try_sympy(m.group(1).strip()), text, flags=re.DOTALL)
    text = re.sub(r"\$(.*?)\$", lambda m: try_sympy(m.group(1).strip()), text)
    return text


def strip_prompt_markers(text: str) -> str:
    lines = text.splitlines()
    clean = [l for l in lines if not l.strip().startswith(">>") and not l.startswith("[INFO]")]
    return "\n".join(clean).strip()


class LoadingSpinner(Static):
    """Custom animated loading spinner."""
    SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spinner_index = 0
        self.update(f"[bold #f0a500]{self.SPINNERS[0]} Loading...")
    
    def on_mount(self):
        self.set_interval(0.1, self.update_spinner)
    
    def update_spinner(self):
        self.spinner_index = (self.spinner_index + 1) % len(self.SPINNERS)
        self.update(f"[bold #f0a500]{self.SPINNERS[self.spinner_index]} Loading...")


class BrainButton(Static):
    """Compact brain toggle button that doesn't steal focus."""
    
    def __init__(self, **kwargs):
        super().__init__("🧠", **kwargs)
        self.can_focus = False
    
    @on(Click)
    def handle_click(self, event: Click) -> None:
        self.app.thinking_enabled = not self.app.thinking_enabled
        self.set_class(self.app.thinking_enabled, "active")
        self.app.query_one("#input", ChatInput).focus()


class ChatInput(TextArea):
    async def _on_key(self, event: Key) -> None:
        if event.key is None:
            return

        if event.key == "enter":
            event.prevent_default()
            event.stop()
            await self.app.action_submit()
            return

        if event.key == "ctrl+c":
            event.prevent_default()
            event.stop()
            self.app.exit()
            return

        if event.key == "escape":
            event.prevent_default()
            event.stop()
            await self.app.action_interrupt()
            return

        await super()._on_key(event)


class ChatUI(App):
    BINDINGS = [("ctrl+c", "quit", "Quit")]
    CSS = """
    Screen {
        layout: vertical;
        background: #0f0f0f;
    }

    #splash-container {
        layout: vertical;
        width: 100%;
        height: 100%;
        align: center middle;
    }

    #splash-logo {
        text-align: center;
        color: #f0a500;
        margin-bottom: 1;
    }

    #load-spinner {
        width: 1fr;
        border: none;
        text-align: center;
    }

    #chat-center {
        height: 1fr;
        width: 100%;
        align: center top;
        display: none;
    }

    #chat {
        height: 100%;
        width: 88;
        padding: 2;
        layout: vertical;
        align: center top;
    }

    .bubble-user {
        margin-top: 1;
        padding: 1 2;
        background: #1a1a1a;
        border: round #282828;
        color: #d8d8d8;
    }

     .bubble-assistant {
         margin-bottom: 1;
         padding: 0 2;
         color: #f0a500;
     }

    .bubble-welcome {
        margin-bottom: 1;
        padding: 0 2;
        color: #7a7a7a;
        text-align: center;
        width: 100%;
    }

    #input-center {
        width: 100%;
        align: center bottom;
        padding-bottom: 1;
        display: none;
    }

    #input-card {
        width: 88;
        background: #161616;
        border: round #252525;
        height: 3;
        layout: horizontal;
    }

    #input {
        background: #161616;
        color: #e0e0e0;
        border: none;
        width: 1fr;
        margin: 0 1;
    }

    #send-btn {
        width: 8;
        background: #f0a500;
        color: #000;
        text-style: bold;
        text-align: center;
    }

     #send-btn.stopping {
          background: #e05a5a;
          color: #fff;
      }

     #brain-btn {
         width: 3;
         min-width: 3;
         padding: 0;
         border: none;
         background: transparent;
         color: #444;
         text-align: center;
         content-align: center middle;
     }

     #brain-btn.active {
         color: #f0a500;
     }

     #brain-btn:hover {
         background: #252525;
     }

     #brain-btn:focus {
         background: transparent;
         border: none;
     }

      .bubble-prompt {
         margin: 3 0;
         padding: 3;
         width: 100%;
         color: #f0a500;
         text-style: bold;
         height: auto;
     }
     """

    def compose(self) -> ComposeResult:
        with Center(id="splash-container"):
            yield Static(LOGO, id="splash-logo")
            yield LoadingSpinner(id="load-spinner")

        with Vertical(id="chat-center"):
            yield VerticalScroll(id="chat")

        with Center(id="input-center"):
            with Horizontal(id="input-card"):
                yield BrainButton(id="brain-btn", classes="brain-btn")
                yield ChatInput(id="input")
                yield Static(" SEND ", id="send-btn")

    async def on_mount(self):
        self.busy = False
        self.interrupted = False
        self.first_message = True
        self.thinking_enabled = False
        asyncio.create_task(self.initialize_model())

    async def initialize_model(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        self.proc = await asyncio.create_subprocess_exec(
            *BASE_CMD,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )

        await self._read_until_prompt()

        self.query_one("#splash-container").display = False
        self.query_one("#chat-center").display = True
        self.query_one("#input-center").display = True
        
        # Show random welcome message in chat, centered
        chat = self.query_one("#chat", VerticalScroll)
        welcome = random.choice(WELCOME_MESSAGES)
        await chat.mount(Markdown(f"```\n{welcome}\n```", classes="bubble-welcome"))
        
        # Show prompt message
        await chat.mount(Static("How can I help you?", classes="bubble-prompt"))
        
        chat.scroll_end(animate=False)
        
        self.query_one("#input").focus()

    async def action_submit(self):
        if self.busy:
            return

        box = self.query_one("#input", ChatInput)
        user_text = box.text.strip()
        if not user_text:
            return

        # Prefix with /think if brain toggle is enabled
        if self.thinking_enabled:
            user_text = "/think " + user_text

        box.clear()

        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(Markdown(user_text, classes="bubble-user"))

        self.current_md = Markdown("▌", classes="bubble-assistant")
        await chat.mount(self.current_md)
        chat.scroll_end(animate=False)

        self._set_busy(True)
        asyncio.create_task(self.run_model(user_text))

    def _set_busy(self, busy: bool):
        self.busy = busy
        btn = self.query_one("#send-btn", Static)
        btn.update(" STOP " if busy else " SEND ")
        btn.set_class(busy, "stopping")

    async def on_static_click(self, event: Click):
        if event.widget.id == "send-btn":
            if self.busy:
                await self.action_interrupt()
            else:
                await self.action_submit()

    async def action_interrupt(self):
        if self.busy:
            self.proc.stdin.write(b"\x04")
            await self.proc.stdin.drain()
            self.interrupted = True

    async def _read_until_prompt(self):
        buf = ""
        while True:
            chunk = await self.proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk.decode(errors="ignore")
            if buf.endswith(">> "):
                break
        return buf

    async def run_model(self, user_text: str):
        # Add extra delay for first message to ensure model is ready
        if self.first_message:
            await asyncio.sleep(2)
            self.first_message = False

        self.proc.stdin.write((user_text + "\n").encode())
        await self.proc.stdin.drain()

        buf = ""
        last_update = 0
        thinking_spinner = None
        chat = self.query_one("#chat", VerticalScroll)
        in_thinking = False

        def is_thinking(buffer):
            """Check if we're currently inside a thinking block."""
            start_count = buffer.count("<think>")
            end_count = buffer.count("</think>")
            return start_count > end_count

        def get_display_text(buffer):
            """Extract display text (after thinking blocks)."""
            # Find the last occurrence of </think>
            last_end = buffer.rfind("</think>")
            
            if last_end >= 0:
                # Return text after the last </think> tag
                return buffer[last_end + len("</think>"):].strip()
            elif "<think>" in buffer:
                # We're inside a thinking block, no display text yet
                return ""
            
            # No thinking tags, return buffer as-is
            return buffer.strip()

        while True:
            chunk = await self.proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk.decode(errors="ignore")

            if buf.endswith(">> "):
                break

            # Check thinking state
            currently_thinking = is_thinking(buf)

            if currently_thinking and not in_thinking:
                # Thinking just started
                in_thinking = True
                thinking_spinner = LoadingSpinner()
                await chat.mount(thinking_spinner)
                # Clear the current markdown to hide thinking text
                await self.current_md.update("")
            elif not currently_thinking and in_thinking:
                # Thinking just ended
                in_thinking = False
                if thinking_spinner:
                    await thinking_spinner.remove()
                    thinking_spinner = None

            # Update display if not thinking
            if not in_thinking:
                now = asyncio.get_event_loop().time()
                if now - last_update > 0.05:
                    display = strip_prompt_markers(transform_math(get_display_text(buf)))
                    await self.current_md.update(f"{display} ▌")
                    chat.scroll_end(animate=False)
                    last_update = now

        # Final update
        display = strip_prompt_markers(transform_math(get_display_text(buf)))
        if self.interrupted:
            display += "\n\n*— stopped*"
            self.interrupted = False

        # Clean up spinner if still active
        if thinking_spinner:
            await thinking_spinner.remove()

        await self.current_md.update(display)
        chat.scroll_end(animate=False)
        self._set_busy(False)


if __name__ == "__main__":
    ChatUI().run()
