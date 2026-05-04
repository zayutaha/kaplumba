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
from textual.containers import VerticalScroll, Vertical, Horizontal, Center, Middle
from textual.events import Key, Click
from pylatexenc.latex2text import LatexNodes2Text
from sympy import sympify, pretty

SYSTEM_PROMPT = """AI PERSONA AND STYLE GUIDELINES

CORE TONE AND VOICE

Neutrality: Maintain a calm, objective, and level-headed tone. When addressing complex, sensitive, or hard themes, avoid moralizing, over-explaining, or using flowery, empathetic fillers. Deliver insights directly and dispassionately.

Human-Centric Flow: Write like a person in a professional or academic setting. Use natural sentence variety and focus on a logical progression of thought rather than a customer service persona.

Minimalist Punctuation: Do not use em-dashes unless they are grammatically essential for clarity. Avoid using them to create vibe or pause for dramatic effect.

STRUCTURAL CONSTRAINTS

Paragraph-First Delivery: Prioritize well-constructed paragraphs over formatting tools. Avoid lists, bold headers, and nested blocks unless the data is strictly comparative or technical.

Steerability: Be highly responsive to specific instructions regarding tone or subject matter depth. If a user pushes into difficult or controversial territory, provide the requested information or analysis without safety-canned hedging, focusing instead on objective facts and diverse perspectives.

ENGAGEMENT RULES

Directness: Eliminate introductory phrases like "As an AI" or "It is important to remember" and concluding summaries that restate what has already been said.

Nuance over Certainty: Acknowledge complexity where it exists without using clichés. If a topic is hard or lacks a clear answer, describe the tension of the subject matter rather than defaulting to a neutral middle-ground."""
BASE_CMD = [
    "uv", "run", "python", "-m", "mlx_lm.chat",
    "--model", "/Users/zayaantaha/.omlx/models/SHHHQwen",
    "--temp", "0.7",
    "--top-p", "0.8",
    "--max-tokens", "16384",
    "--xtc-probability", "0.0",
    "--xtc-threshold", "0.0",
    "--mtp",
    "--turbo-kv-bits", "3",
    "--turbo-fp16-layers", "2",
    "--chat-template-args", '{"enable_thinking":false}',
    "--system-prompt", SYSTEM_PROMPT,
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

    def __init__(self, message="Loading model...", **kwargs):
        super().__init__(**kwargs)
        self.spinner_index = 0
        self.message = message
        self.update(f"[bold #f0a500]{self.SPINNERS[0]} {self.message}")

    def on_mount(self):
        self.set_interval(0.1, self.update_spinner)

    def update_spinner(self):
        self.spinner_index = (self.spinner_index + 1) % len(self.SPINNERS)
        self.update(f"[bold #f0a500]{self.SPINNERS[self.spinner_index]} {self.message}")


class ChatInput(TextArea):
    def on_mount(self) -> None:
        """Initialize the input."""
        self.show_line_numbers = False
        self.soft_wrap = True
        self.styles.height = 1
        self.set_interval(0.05, self.sync_height)

    def sync_height(self) -> None:
        """Sync widget height to content height (including wrapped lines)."""
        target_height = min(max(1, self.virtual_size.height), 5)
        current = self.styles.height
        if current is None or getattr(current, 'value', current) != target_height:
            self.styles.height = target_height
            self.refresh()

    async def _on_key(self, event: Key) -> None:
        if event.key is None:
            return

        # Don't capture keys if crash dialog is visible
        if self.app.crash_dialog_visible:
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
    padding: 1 2 0 2;
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
    height: auto;
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
    content-align: center middle;
    height: 100%;
}

#send-btn.stopping {
    background: #e05a5a;
    color: #fff;
}

.bubble-prompt {
    margin: 3 0;
    padding: 3;
    width: 100%;
    color: #f0a500;
    text-style: bold;
    height: auto;
}

#crash-dialog-container {
    layout: vertical;
    width: 100%;
    height: 100%;
    align: center middle;
    display: none;
    background: rgba(0, 0, 0, 0.7);
}

#crash-dialog {
    width: 40;
    height: auto;
    background: #1a1a1a;
    border: round #f0a500;
    padding: 2;
    align: center middle;
}

.crash-message {
    color: #f0a500;
    text-align: center;
    margin-bottom: 1;
}

.crash-buttons {
    align: center middle;
    height: auto;
}

.crash-buttons Button {
    margin: 0 1;
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
                yield ChatInput(id="input")
                yield Static(" SEND ", id="send-btn")

        with Middle(id="crash-dialog-container"):
            with Vertical(id="crash-dialog", classes="crash-dialog"):
                yield Static("Model crashed. What do you want to do?", id="crash-message", classes="crash-message")
                with Horizontal(classes="crash-buttons"):
                    yield Button("Reload", id="crash-reload", variant="primary")
                    yield Button("Quit", id="crash-quit", variant="error")

    async def on_mount(self):
        self.busy = False
        self.interrupted = False
        self.loading = False
        self.first_message = True
        self.reloading = False
        self.crash_count = 0
        self.max_crashes = 3
        self.crash_dialog_visible = False
        asyncio.create_task(self.initialize_model())

    async def initialize_model(self):
        self.loading = True
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        self.proc = await asyncio.create_subprocess_exec(
            *BASE_CMD,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )

        buf = await self._read_until_prompt()

        if not buf.endswith(">> "):
            await self._handle_crash("Model failed to initialize")
            return

        # Update spinner to show warming up phase
        spinner = self.query_one("#load-spinner", LoadingSpinner)
        spinner.message = "Warming up..."
        spinner.spinner_index = 0
        spinner.update(f"[bold #f0a500]{spinner.SPINNERS[0]} {spinner.message}")

        # Warm-up: send dummy message to verify model works
        try:
            self.proc.stdin.write(b"warmup\n")
            await self.proc.stdin.drain()
        except Exception:
            await self._handle_crash("")
            return

        # Wait for warm-up response
        buf = ""
        while True:
            try:
                chunk = await self.proc.stdout.read(256)
            except Exception:
                await self._handle_crash("")
                return
            if not chunk:
                await self._handle_crash("")
                return
            buf += chunk.decode(errors="ignore")
            if buf.endswith(">> "):
                break

        # Reset conversation so user doesn't see warm-up
        try:
            self.proc.stdin.write(b"r\n")
            await self.proc.stdin.drain()
        except Exception:
            pass

        # Wait for reset to complete
        buf = await self._read_until_prompt()
        if not buf.endswith(">> "):
            await self._handle_crash("")
            return

        self.crash_count = 0
        self._show_chat_ui()

    def _show_chat_ui(self):
        self.loading = False
        self.query_one("#splash-container").display = False
        self.query_one("#chat-center").display = True
        self.query_one("#input-center").display = True

        if self.reloading:
            self.reloading = False
            self.query_one("#input").focus()
            return

        chat = self.query_one("#chat", VerticalScroll)
        welcome = random.choice(WELCOME_MESSAGES)
        chat.mount(Markdown(f"```\n{welcome}\n```", classes="bubble-welcome"))
        chat.mount(Static("How can I help you?", classes="bubble-prompt"))
        chat.scroll_end(animate=False)
        self.query_one("#input").focus()

    def _show_loading_ui(self, message="Loading model..."):
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        splash = self.query_one("#splash-container")
        splash.display = True
        spinner = self.query_one("#load-spinner", LoadingSpinner)
        spinner.message = message
        spinner.spinner_index = 0
        spinner.update(f"[bold #f0a500]{spinner.SPINNERS[0]} {spinner.message}")

    async def action_submit(self):
        if self.busy or self.loading or not self.proc or self.proc.returncode is not None:
            return

        box = self.query_one("#input", ChatInput)
        user_text = box.text.strip()
        if not user_text:
            return

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

    async def _handle_crash(self, error_msg):
        """Handle model crash: show dialog with quit/reload options."""
        self._set_busy(False)
        self.loading = True
        self.crash_count += 1

        if self.crash_count >= self.max_crashes:
            self.exit("Too many crashes, giving up")
            return

        # If still initializing (no chat UI yet), auto-reload without dialog
        if self.query_one("#chat-center").display == False:
            self.reloading = True
            self._show_loading_ui(f"Reloading model (crash #{self.crash_count})...")
            if self.proc and self.proc.returncode is None:
                try:
                    self.proc.kill()
                    await self.proc.wait()
                except Exception:
                    pass
            asyncio.create_task(self.initialize_model())
            return

        # Show crash dialog for runtime crashes
        self.reloading = True
        self.crash_dialog_visible = True
        self.query_one("#crash-dialog-container").display = True
        self.query_one("#crash-message").update(f"Model crashed (attempt {self.crash_count}/{self.max_crashes}). Reload or quit?")
        self.query_one("#crash-reload").focus()

    async def on_key(self, event: Key) -> None:
        """Handle key presses globally."""
        # If crash dialog is visible and Enter is pressed, trigger reload
        if event.key == "enter" and self.crash_dialog_visible:
            self.query_one("#crash-reload").press()
            event.prevent_default()
            event.stop()

    async def on_button_pressed(self, event: Button.Pressed):
        """Handle crash dialog button presses."""
        if event.button.id == "crash-reload":
            self.crash_dialog_visible = False
            self.query_one("#crash-dialog-container").display = False
            self._show_loading_ui(f"Reloading model (crash #{self.crash_count})...")
            if self.proc and self.proc.returncode is None:
                try:
                    self.proc.kill()
                    await self.proc.wait()
                except Exception:
                    pass
            asyncio.create_task(self.initialize_model())
        elif event.button.id == "crash-quit":
            self.exit("Model crashed")

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
        if self.first_message:
            await asyncio.sleep(2)
            self.first_message = False

        if not self.proc or self.proc.returncode is not None:
            await self._handle_crash("")
            return

        user_text = " ".join(user_text.split("\n"))

        try:
            self.proc.stdin.write((user_text + "\n").encode())
            await self.proc.stdin.drain()
        except Exception:
            await self._handle_crash("")
            return

        buf = ""
        last_update = 0
        chat = self.query_one("#chat", VerticalScroll)
        thinking_enabled = user_text.startswith("/think")

        def get_display_text(buffer):
            last_end = buffer.rfind("</think>")
            if last_end >= 0:
                return buffer[last_end + len("</think>"):].strip()
            return ""

        spinner_index = 0
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        while True:
            try:
                chunk = await self.proc.stdout.read(256)
            except Exception:
                await self._handle_crash("")
                return

            if not chunk:
                await self._handle_crash("")
                return

            buf += chunk.decode(errors="ignore")

            if buf.endswith(">> "):
                break

            now = asyncio.get_event_loop().time()
            if now - last_update > 0.05:
                if thinking_enabled:
                    if "</think>" not in buf:
                        spinner_index = (spinner_index + 1) % len(spinner_frames)
                        await self.current_md.update(f"Thinking... {spinner_frames[spinner_index]}")
                    else:
                        display = strip_prompt_markers(transform_math(get_display_text(buf)))
                        if display:
                            await self.current_md.update(f"{display} ▌")
                else:
                    display = strip_prompt_markers(transform_math(buf))
                    await self.current_md.update(f"{display} ▌")

                last_update = now

        if thinking_enabled:
            display = strip_prompt_markers(transform_math(get_display_text(buf)))
        else:
            display = strip_prompt_markers(transform_math(buf))

        if self.interrupted:
            display += "\n\n*— stopped*"
            self.interrupted = False

        await self.current_md.update(display)
        # Only scroll to bottom on completion if user is near bottom
        scroll_y = chat.scroll_offset.y
        virtual_h = chat.virtual_size.height
        widget_h = chat.region.height
        if virtual_h <= widget_h:
            chat.scroll_end(animate=False)
        else:
            max_scroll_y = virtual_h - widget_h
            if max_scroll_y - scroll_y <= 50:
                chat.scroll_end(animate=False)
        self._set_busy(False)


if __name__ == "__main__":
    ChatUI().run()
