# 🔴 MUST be first — fixes Textual crash
import textual.keys as tk

_orig = tk.key_to_character
def safe_key_to_character(key):
    if key is None:
        return None
    return _orig(key)
tk.key_to_character = safe_key_to_character


import random
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Center, Horizontal, Middle, Vertical, VerticalScroll
from textual.events import Click, Key
from textual.widgets import Button, Markdown, Static

from model_catalog import list_models
from textual_ui.styles import CHAT_CSS, LOGO, WELCOME_MESSAGES
from textual_ui.latex import format_for_display, strip_prompt_markers
from textual_ui.personas import PERSONALITIES

from textual_ui.widgets.chat_input import ChatInput
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
                    ("default", "Blunt, compact answers with no fake politeness."),
                    ("doctor", "Medical explainer who asks follow-up questions first."),
                    ("historian", "Opinionated historical analysis with sharper language."),
                ],
                id="personality-selector",
            )

        with Center(id="options-selector-container"):
            yield OptionsSelector({}, id="options-selector")

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
        box.load_text(selected)
        self.hide_command_menu()

    def _mount_welcome_screen(self) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        existing_ids = {child.id for child in chat.children}
        if "welcome-logo" in existing_ids or "welcome-prompt" in existing_ids:
            return
        welcome = random.choice(WELCOME_MESSAGES)
        chat.mount(Markdown(f"```\n{welcome}\n```", id="welcome-logo", classes="bubble-welcome"))
        chat.mount(Static("How can I help you?", id="welcome-prompt", classes="bubble-prompt"))
        chat.scroll_end(animate=False)

    # ── Action handlers ──

    async def action_submit(self):
        await self.controller.handle_submit()

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
        await chat.mount(Markdown(user_text, classes="bubble-user"))
        self._stream_generation += 1
        self.current_md = Markdown("▌", classes="bubble-assistant")
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

    def show_model_loading(self, message="Loading model..."):
        self.show_loading(message)

    def hide_model_loading(self):
        self.loading = False
        self.query_one("#splash-container").display = False
