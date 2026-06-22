# 🔴 MUST be first — fixes Textual crash
import textual.keys as tk

_orig = tk.key_to_character
def safe_key_to_character(key):
    if key is None:
        return None
    return _orig(key)
tk.key_to_character = safe_key_to_character


import asyncio
import random
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Center, Horizontal, Middle, Vertical, VerticalScroll
from textual.events import Click, Key
from textual.widgets import Button, Markdown, Static
from textual import on


_selected_bubbles: list["CopyableMarkdown"] = []

class CopyableMarkdown(Markdown):
    """Pass-through markdown. No click handling — terminal handles text selection natively."""


async def _copy_selected(app: "ChatUI"):
    if not _selected_bubbles:
        app.notify("No bubbles selected — click to select", timeout=1.5)
        return
    parts = []
    for b in _selected_bubbles:
        t = b._markdown if hasattr(b, '_markdown') and b._markdown else b._initial_markdown or ""
        if t:
            parts.append(t)
    if not parts:
        return
    text = "\n\n".join(parts)
    try:
        proc = await asyncio.create_subprocess_exec(
            "pbcopy", stdin=asyncio.subprocess.PIPE,
        )
        await proc.communicate(input=text.encode())
        if proc.returncode == 0:
            msg = f"Copied {len(parts)} message{'' if len(parts) == 1 else 's'} to clipboard"
            app.notify(msg, timeout=2)
        else:
            app.notify("Failed to copy", severity="error", timeout=1.5)
    except FileNotFoundError:
        app.notify("pbcopy not available", severity="error", timeout=1.5)
    # Clear selections
    for b in _selected_bubbles:
        b.remove_class("bubble-selected")
    _selected_bubbles.clear()
    app._update_selection_ui()

from model_catalog import list_models
from textual_ui.styles import CHAT_CSS, LOGO, WELCOME_MESSAGES

HELP_TEXT = """\
[bright_white]Help[/]

[bold]Chat[/]
  Type a message and press Enter to send
  [bold]Ctrl+C[/]     Quit
  [bold]Ctrl+R[/]     Reload model

[bold]Copy messages[/]
  [bold]V[/]           Toggle select mode
            (click bubbles to mark them)
  [bold]C[/]           Copy all marked messages
  [bold]Esc[/]         Exit select mode

[bold]Copy text[/]
  [bold]Shift+drag[/]  Select text with mouse
  [bold]Cmd+C[/]       Copy selected text

[bold]Commands[/]
  [bold]/clear[/]      Reset conversation
  [bold]/models[/]     Switch model
  [bold]/options[/]    Change model options
  [bold]/memory[/]     Show GPU memory
  [bold]/personality[/] Change personality

[dim]Press [bold]Ctrl+\\[/] or Esc or click outside to close[/]"""
from textual_ui.latex import format_for_display, strip_prompt_markers
from textual_ui.personas import PERSONALITY_INFO
from settings_store import NO_ARG_COMMANDS
from textual.widgets._text_area import Selection

from textual_ui.widgets.chat_input import ChatInput
from textual_ui.widgets.chat_selector import ChatSelector
from textual_ui.widgets.loading_spinner import LoadingSpinner
from textual_ui.widgets.model_picker import ModelSelector
from textual_ui.widgets.options_selector import OptionsSelector
from textual_ui.widgets.personality_selector import PersonalitySelector
from textual_ui.widgets.slash_command_menu import SlashCommandMenu
from orchestrator import Orchestrator


class ChatUI(App):
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "reload_model", "Reload Model"),
        ("v", "toggle_select_mode", "Select"),
        ("c", "copy_selected", "Copy"),
        ("ctrl+backslash", "show_help", "Help"),
        ("escape", "close_help", "Close"),
    ]
    CSS = CHAT_CSS

    def __init__(self, port=None, on_crash=None, on_reload=None, on_quit=None, **kwargs):
        super().__init__(**kwargs)
        self.controller = Orchestrator(self, port=port)
        self._on_crash = on_crash or self.controller.handle_crash_from_chat
        self._on_reload = on_reload or self.controller.handle_reload
        self._on_quit = on_quit or self.controller.handle_quit
        self.loading = False
        self.busy = False
        self.interrupted = False
        self.select_mode = False
        self.first_message = True
        self.crash_dialog_visible = False
        self.current_md = None
        self._stream_generation = 0

    def compose(self) -> ComposeResult:
        with Center(id="model-selector-container"):
            yield ModelSelector(list_models({}), id="model-selector")

        with Center(id="personality-selector-container"):
            yield PersonalitySelector(
                [
                    (name, info["description"])
                    for name, info in PERSONALITY_INFO.items()
                ],
                id="personality-selector",
            )

        with Center(id="options-selector-container"):
            yield OptionsSelector({}, id="options-selector")

        with Center(id="chat-selector-container"):
            yield ChatSelector(id="chat-selector")

        with Center(id="splash-container"):
            yield Static(LOGO, id="splash-logo")
            yield LoadingSpinner(id="load-spinner")

        with Vertical(id="chat-center"):
            yield VerticalScroll(id="chat")

        with Center(id="command-menu-container"):
            yield SlashCommandMenu(id="command-menu")

        with Center(id="input-center"):
            with Horizontal(id="input-card"):
                yield ChatInput(id="input")
                yield Static(" SEND ", id="send-btn")

        with Middle(id="help-overlay"):
            with Vertical(id="help-box", classes="help-box"):
                yield Static("", id="help-content")

        with Middle(id="crash-dialog-container"):
            with Vertical(id="crash-dialog", classes="crash-dialog"):
                yield Static("Model crashed. What do you want to do?", id="crash-message", classes="crash-message")
                with Horizontal(classes="crash-buttons"):
                    yield Button("Reload", id="crash-reload", variant="primary")
                    yield Button("Quit", id="crash-quit", variant="error")

    async def on_mount(self):
        self.loading = False
        self.busy = False
        self.interrupted = False
        self.first_message = True
        self.crash_dialog_visible = False
        self.query_one("#splash-container").display = False
        self.query_one("#model-selector-container").display = True
        self.query_one("#personality-selector-container").display = False
        self.query_one("#options-selector-container").display = False
        self.query_one("#chat-selector-container").display = False
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#model-selector").focus()

    # ── Public UI control methods (called by orchestrator) ──

    def show_chat_ui(self):
        self.loading = False
        self.query_one("#splash-container").display = False
        self.query_one("#model-selector-container").display = False
        self.query_one("#personality-selector-container").display = False
        self.query_one("#options-selector-container").display = False
        self.query_one("#chat-selector-container").display = False
        self.query_one("#chat-center").display = True
        self.query_one("#input-center").display = True
        self._mount_welcome_screen()
        self.refresh_command_menu()
        self.query_one("#input").focus()

    def show_loading(self, message="Loading model..."):
        self.loading = True
        self.query_one("#model-selector-container").display = False
        self.query_one("#personality-selector-container").display = False
        self.query_one("#options-selector-container").display = False
        self.query_one("#chat-selector-container").display = False
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        splash = self.query_one("#splash-container")
        splash.display = True
        spinner = self.query_one("#load-spinner", LoadingSpinner)
        spinner.message = message
        spinner.spinner_index = 0
        spinner.update(f"[bold #f0a500]{spinner.SPINNERS[0]} {spinner.message}")

    async def reset_chat(self):
        await self.clear_chat()
        self._mount_welcome_screen()

    async def clear_chat(self):
        chat = self.query_one("#chat", VerticalScroll)
        for child in list(chat.children):
            await child.remove()
        _selected_bubbles.clear()
        self._update_selection_ui()

    def _update_selection_ui(self):
        btn = self.query_one("#send-btn", Static)
        n = len(_selected_bubbles)
        if self.select_mode:
            label = f" SELECT {n} " if n else " SELECT "
            btn.update(label)
            btn.set_class(self.busy, "stopping")
        elif self.busy:
            btn.update(" STOP ")
            btn.set_class(True, "stopping")
        else:
            btn.update(" SEND ")
            btn.set_class(False, "stopping")

    async def action_close_help(self):
        box = self.query_one("#help-overlay")
        if box.display:
            box.display = False
            self.query_one("#chat-center").display = True
            self.query_one("#input-center").display = True

    async def show_overlay(self, title: str, body: str):
        """Show the help overlay with custom content."""
        content = self.query_one("#help-content", Static)
        content.update(f"[bold]{title}[/]\n\n{body}\n\n[dim]Esc or click outside to close[/]")
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#help-overlay").display = True

    async def action_show_help(self):
        box = self.query_one("#help-overlay")
        if not box.display:
            content = self.query_one("#help-content", Static)
            content.update(HELP_TEXT)
            self.query_one("#chat-center").display = False
            self.query_one("#input-center").display = False
        else:
            self.query_one("#chat-center").display = True
            self.query_one("#input-center").display = True
        box.display = not box.display

    async def action_copy_selected(self):
        if not _selected_bubbles:
            self.notify("Press V to enter select mode, click bubble to mark, C to copy", timeout=3)
            return
        await _copy_selected(self)

    # ── Chat UI internals ──

    @property
    def command_menu_visible(self) -> bool:
        return bool(self.query_one("#command-menu-container").display)

    def refresh_command_menu(self) -> None:
        if self.loading or self.busy or not self.query_one("#chat-center").display or not self.query_one("#input-center").display:
            self.query_one("#command-menu-container").display = False
            return
        box = self.query_one("#input", ChatInput)
        container = self.query_one("#command-menu-container")
        menu = self.query_one("#command-menu", SlashCommandMenu)
        container.display = menu.update_matches(box.text)

    def hide_command_menu(self) -> None:
        menu = self.query_one("#command-menu", SlashCommandMenu)
        menu.matches = []
        menu.selected_index = 0
        self.query_one("#command-menu-container").display = False

    def move_command_selection(self, direction: int) -> None:
        menu = self.query_one("#command-menu", SlashCommandMenu)
        if menu.move_selection(direction):
            self.query_one("#command-menu-container").display = True

    def apply_selected_command(self) -> None:
        menu = self.query_one("#command-menu", SlashCommandMenu)
        selected = menu.selected_command()
        if not selected:
            return
        box = self.query_one("#input", ChatInput)
        if selected in NO_ARG_COMMANDS:
            box.load_text(selected)
            self.hide_command_menu()
            asyncio.create_task(self.action_submit())
        else:
            text = selected + " "
            box.load_text(text)
            box.selection = Selection(start=(0, len(text)), end=(0, len(text)))
            self.query_one("#command-menu-container").display = True

    def _mount_welcome_screen(self) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        existing_ids = {child.id for child in chat.children}
        if "welcome-logo" in existing_ids or "welcome-prompt" in existing_ids:
            return
        welcome = random.choice(WELCOME_MESSAGES)
        chat.mount(CopyableMarkdown(f"```\n{welcome}\n```", id="welcome-logo", classes="bubble-welcome"))
        chat.mount(Static("How can I help you?", id="welcome-prompt", classes="bubble-prompt"))
        chat.scroll_end(animate=False)

    # ── Action handlers ──

    async def action_toggle_select_mode(self):
        self.select_mode = not self.select_mode
        self._update_selection_ui()
        if self.select_mode:
            self.notify("SELECT: click bubble to mark, C to copy all", timeout=3)
        else:
            self.notify("Shift+drag to select text, Cmd+C to copy", timeout=3)

    async def action_submit(self):
        await self.controller.handle_submit()

    def _set_busy(self, busy: bool):
        self.busy = busy
        if busy:
            self.query_one("#send-btn", Static).update(" STOP ")
        else:
            self._update_selection_ui()

    @on(Click, "#send-btn")
    async def on_send_click(self):
        if _selected_bubbles:
            await _copy_selected(self)
        elif self.busy:
            await self.controller.handle_interrupt()
        else:
            await self.controller.handle_submit()

    async def on_click(self, event: Click):
        help_overlay = self.query_one("#help-overlay")
        if help_overlay.display:
            help_overlay.display = False
            self.query_one("#chat-center").display = True
            self.query_one("#input-center").display = True
            return
        if not self.select_mode:
            return
        widget = event.widget
        if isinstance(widget, CopyableMarkdown):
            text = widget._markdown if hasattr(widget, '_markdown') and widget._markdown else widget._initial_markdown or ""
            if not text:
                return
            if widget in _selected_bubbles:
                _selected_bubbles.remove(widget)
                widget.remove_class("bubble-selected")
            else:
                _selected_bubbles.append(widget)
                widget.add_class("bubble-selected")
            self._update_selection_ui()
            event.stop()

    async def action_interrupt(self):
        await self.controller.handle_interrupt()

    async def action_quit(self):
        await self.controller.handle_quit()

    async def on_key(self, event: Key) -> None:
        if event.key == "enter" and self.crash_dialog_visible:
            self.query_one("#crash-reload").press()
            event.prevent_default()
            event.stop()

    async def action_reload_model(self) -> None:
        await self.controller.handle_reload()

    async def action_model_selected(self, model_name: str) -> None:
        await self.controller.handle_model_selected(model_name)

    async def action_model_edit(self, model_name: str) -> None:
        config = self.controller.get_model_config(model_name)
        sel = self.query_one("#options-selector", OptionsSelector)
        sel.set_editor_mode(model_name=model_name, options=config.get("options", {}), personality=config.get("personality", "default"))
        self.query_one("#options-selector-container").display = True
        self.query_one("#model-selector-container").display = False
        self.query_one("#personality-selector-container").display = False
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        sel.focus()

    async def action_model_editor_save(self, model_name: str, config: dict) -> None:
        await self.controller.handle_model_config_saved(model_name, config)
        self.query_one("#options-selector-container").display = False
        self.query_one("#model-selector-container").display = True
        self.query_one("#model-selector", ModelSelector).focus()

    async def action_options_selected(self, options: dict) -> None:
        await self.controller.handle_options_selected(options)

    async def action_dismiss_options_selector(self) -> None:
        if self.controller.port.running:
            self.show_chat_ui()
            return
        self.query_one("#options-selector-container").display = False
        self.query_one("#model-selector-container").display = True
        self.query_one("#model-selector", ModelSelector).focus()

    async def action_personality_selected(self, personality: str) -> None:
        await self.controller.handle_personality_selected(personality)

    async def action_dismiss_personality_selector(self) -> None:
        if self.controller.port.running:
            self.show_chat_ui()
            return
        self.query_one("#personality-selector-container").display = False
        self.query_one("#model-selector-container").display = True
        self.query_one("#model-selector", ModelSelector).focus()

    async def action_dismiss_model_selector(self) -> None:
        if self.controller.port.running:
            self.show_chat_ui()
            return
        self.query_one("#model-selector-container").display = True
        self.query_one("#model-selector", ModelSelector).focus()

    async def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "crash-reload":
            await self.controller.handle_reload()
        elif event.button.id == "crash-quit":
            await self.controller.handle_quit()

    async def action_chat_selected(self, chat_name: str) -> None:
        await self.controller.handle_chat_selected(chat_name)

    async def action_dismiss_chat_selector(self) -> None:
        if self.controller.port.running:
            self.show_chat_ui()
            return
        self.query_one("#chat-selector-container").display = False
        self.query_one("#model-selector-container").display = True
        self.query_one("#model-selector", ModelSelector).focus()

    async def show_chat_selector(self):
        self.query_one("#chat-selector", ChatSelector).refresh_chats()
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#model-selector-container").display = False
        self.query_one("#personality-selector-container").display = False
        self.query_one("#options-selector-container").display = False
        self.query_one("#chat-selector-container").display = True
        self.query_one("#chat-selector").focus()

    async def show_model_selector(self):
        self.query_one("#splash-container").display = False
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#personality-selector-container").display = False
        self.query_one("#options-selector-container").display = False
        self.query_one("#model-selector-container").display = True
        self.query_one("#model-selector").focus()

    async def show_options_selector(self):
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#model-selector-container").display = False
        self.query_one("#personality-selector-container").display = False
        self.query_one("#options-selector", OptionsSelector).set_options(
            self.controller.model_options
        )
        self.query_one("#options-selector-container").display = True
        self.query_one("#options-selector").focus()

    async def show_personality_selector(self):
        self.query_one("#splash-container").display = False
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#model-selector-container").display = False
        self.query_one("#options-selector-container").display = False
        self.query_one("#personality-selector-container").display = True
        self.query_one("#personality-selector").focus()

    async def handle_stream_text(self, user_text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(CopyableMarkdown(user_text, classes="bubble-user"))
        self._stream_generation += 1
        self.current_md = CopyableMarkdown("▌", classes="bubble-assistant")
        self.current_md._generation = self._stream_generation
        await chat.mount(self.current_md)
        chat.scroll_end(animate=False)

    async def handle_stream_finished(self, display: str) -> None:
        try:
            if self.current_md and getattr(self.current_md, "_generation", 0) == self._stream_generation:
                await self.current_md.update(display)
        except Exception:
            pass

    async def handle_stream_chunk(self, display: str, show_cursor: bool = True) -> None:
        try:
            if self.current_md and getattr(self.current_md, "_generation", 0) == self._stream_generation:
                cursor = " ▌" if show_cursor else ""
                await self.current_md.update(f"{display}{cursor}")
        except Exception:
            pass

    async def append_user_message(self, text: str):
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(CopyableMarkdown(text, classes="bubble-user"))
        chat.scroll_end(animate=False)

    async def append_assistant_message(self, text: str):
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(CopyableMarkdown(text, classes="bubble-assistant"))
        chat.scroll_end(animate=False)

    def show_model_loading(self, message="Loading model..."):
        self.show_loading(message)

    def hide_model_loading(self):
        self.loading = False
        self.query_one("#splash-container").display = False
