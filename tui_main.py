# 🔴 MUST be first — fixes Textual crash
import textual.keys as tk

_orig = tk.key_to_character
def safe_key_to_character(key):
    if key is None:
        return None
    return _orig(key)
tk.key_to_character = safe_key_to_character


import asyncio
import os
import random
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Static, Button
from textual.containers import VerticalScroll, Vertical, Horizontal, Center, Middle
from textual.events import Key, Click

from tui_commands import BASE_CMD, MODEL_PATH, TUI_PROMPT_MARKER, ModelRunner
from tui_styles import CHAT_CSS, LOGO, WELCOME_MESSAGES
from tui_config import (
    DEFAULT_MODEL_OPTIONS,
    MODEL_CONFIGS_PATH,
    OPTIONS_STATE_PATH,
    load_model_configs,
    load_saved_model_options,
    normalize_model_options,
    save_model_configs,
    save_model_options,
)
from tui_latex import format_for_display, strip_prompt_markers
from tui_personalities import SYSTEM_PROMPT, PERSONALITIES



from tui_loading_spinner import LoadingSpinner
from tui_model_picker import ModelSelector, get_available_models
from tui_personality_selector import PersonalitySelector


from tui_model_config_editor import ModelConfigEditor
from tui_slash_command_menu import SlashCommandMenu


from tui_chat_input import ChatInput
from tui_options_selector import OptionsSelector


class ChatUI(App):
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "reload_model", "Reload Model"),
    ]
    CSS = CHAT_CSS

    def compose(self) -> ComposeResult:
        with Center(id="model-selector-container"):
            yield ModelSelector(get_available_models(DEFAULT_MODEL_OPTIONS), id="model-selector")

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
            yield OptionsSelector(DEFAULT_MODEL_OPTIONS, id="options-selector")

        with Center(id="model-editor-container"):
            yield ModelConfigEditor("", {}, id="model-editor")

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
        self.busy = False
        self.interrupted = False
        self.loading = False
        self.first_message = True
        self.reloading = False
        self.crash_count = 0
        self.max_crashes = 3
        self.crash_dialog_visible = False
        self.selected_model = None
        self.selected_personality = "default"
        self.model_options = load_saved_model_options()
        self.runner = ModelRunner()
        self.query_one("#options-selector", OptionsSelector).set_options(self.model_options)
        self.query_one("#model-selector", ModelSelector).models = get_available_models(self.model_options)
        self.query_one("#model-selector", ModelSelector).render_list()
        self.query_one("#model-selector-container").display = True
        self.query_one("#model-selector").focus()

    def _build_model_command(self, model_path: str) -> list[str]:
        # Merge per-model config overrides
        opts = dict(self.model_options)
        model_name = Path(model_path).name
        configs = load_model_configs()
        model_cfg = configs.get(model_name, {})
        if model_cfg.get("options"):
            opts.update(model_cfg["options"])

        cmd = [
            "uv", "run", "python", "-m", "mlx_lm.chat",
            "--model", model_path,
            "--prompt-marker", TUI_PROMPT_MARKER,
            "--temp", str(opts["temp"]),
            "--top-p", str(opts["top_p"]),
            "--top-k", str(opts["top_k"]),
            "--max-tokens", str(opts["max_tokens"]),
            "--chat-template-args", '{"enable_thinking":false}',
            "--system-prompt", self.current_system_prompt,
        ]
        if opts["mtp"]:
            cmd.append("--mtp")
        if opts["max_kv_size"] is not None:
            cmd.extend(["--max-kv-size", str(opts["max_kv_size"])])
        if opts["turbo_kv_bits"] is not None:
            cmd.extend(["--turbo-kv-bits", str(int(opts["turbo_kv_bits"]))])
        if opts["turbo_fp16_layers"] is not None:
            cmd.extend(["--turbo-fp16-layers", str(opts["turbo_fp16_layers"])])
        return cmd

    async def initialize_model(self):
        self.loading = True
        model_path = str(Path.home() / ".omlx" / "models" / self.selected_model)
        ok = await self.runner.start(model_path, self.model_options, self.current_system_prompt)
        if ok:
            self.crash_count = 0
            self._show_chat_ui()
        else:
            await self._handle_crash("Model failed to initialize")

    def _show_chat_ui(self):
        self.loading = False
        self.query_one("#splash-container").display = False
        self.query_one("#chat-center").display = True
        self.query_one("#input-center").display = True
        self.refresh_command_menu()

        if self.reloading:
            self.reloading = False
            self.query_one("#input").focus()
            return

        self.call_after_refresh(self._mount_welcome_screen)
        self.query_one("#input").focus()

    def _show_loading_ui(self, message="Loading model..."):
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        splash = self.query_one("#splash-container")
        splash.display = True
        spinner = self.query_one("#load-spinner", LoadingSpinner)
        spinner.message = message
        spinner.spinner_index = 0
        spinner.update(f"[bold #f0a500]{spinner.SPINNERS[0]} {spinner.message}")

    @property
    def current_system_prompt(self) -> str:
        return PERSONALITIES.get(self.selected_personality, PERSONALITIES["default"])

    @property
    def command_menu_visible(self) -> bool:
        return bool(self.query_one("#command-menu-container").display)

    def refresh_command_menu(self) -> None:
        if (
            self.loading
            or self.busy
            or self.query_one("#chat-center").display is False
            or self.query_one("#input-center").display is False
        ):
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

    async def _reset_chat_history(self) -> None:
        old_chat = self.query_one("#chat", VerticalScroll)
        await old_chat.remove()
        await self.query_one("#chat-center").mount(VerticalScroll(id="chat"))
        self._mount_welcome_screen()

    async def _clear_chat_only(self) -> None:
        """Clear chat without showing welcome screen (for personality changes mid-conversation)"""
        old_chat = self.query_one("#chat", VerticalScroll)
        await old_chat.remove()
        await self.query_one("#chat-center").mount(VerticalScroll(id="chat"))

    async def _stop_model_process(self) -> None:
        await self.runner.stop()

    async def action_model_selected(self, model_name: str):
        """Handle model selection from the selector screen."""
        self.selected_model = model_name
        # Load per-model personality
        configs = load_model_configs()
        model_cfg = configs.get(model_name, {})
        self.selected_personality = model_cfg.get("personality", "default")
        self.query_one("#model-selector-container").display = False
        self._show_loading_ui(f"Loading {model_name}...")

        await self._reset_chat_history()
        await self._stop_model_process()
        await self.initialize_model()

    async def action_dismiss_model_selector(self):
        """Dismiss model selector and return to chat."""
        self.query_one("#model-selector-container").display = False
        self.query_one("#chat-center").display = True
        self.query_one("#input-center").display = True
        self.refresh_command_menu()
        self.query_one("#input").focus()

    async def show_model_selector(self):
        """Show model selector during chat to switch models."""
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#model-selector-container").display = True
        selector = self.query_one("#model-selector", ModelSelector)
        selector.models = get_available_models(self.model_options)
        if self.selected_model:
            selected_names = [model[0] for model in selector.models]
            selector.selected_index = selected_names.index(self.selected_model) if self.selected_model in selected_names else 0
        else:
            selector.selected_index = 0
        selector.render_list()
        selector.focus()

    async def action_options_selected(self, options: dict[str, object]):
        self.model_options = normalize_model_options(options)
        save_model_options(self.model_options)
        self.query_one("#options-selector-container").display = False
        if not self.selected_model:
            selector = self.query_one("#model-selector", ModelSelector)
            selector.models = get_available_models(self.model_options)
            selector.render_list()
            self.query_one("#chat-center").display = True
            self.query_one("#input-center").display = True
            self.refresh_command_menu()
            self.query_one("#input").focus()
            return

        self.reloading = True
        self._show_loading_ui("Applying options...")
        await self._reset_chat_history()
        await self._stop_model_process()
        await self.initialize_model()

    async def action_dismiss_options_selector(self):
        self.query_one("#options-selector-container").display = False
        self.query_one("#chat-center").display = True
        self.query_one("#input-center").display = True
        self.refresh_command_menu()
        self.query_one("#input").focus()

    async def show_options_selector(self):
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#options-selector-container").display = True
        selector = self.query_one("#options-selector", OptionsSelector)
        selector.set_options(self.model_options)
        selector.focus()

    async def action_model_edit(self, model_name: str):
        configs = load_model_configs()
        config = configs.get(model_name, {})
        editor = self.query_one("#model-editor", ModelConfigEditor)
        editor.model_name = model_name
        editor.options = dict(config.get("options", {}))
        editor.personality = config.get("personality", "default")
        editor._items = editor._build_items()
        editor.selected_index = 0
        editor.render_list()
        self.query_one("#model-selector-container").display = False
        self.query_one("#model-editor-container").display = True
        editor.focus()

    async def action_model_editor_save(self, config: dict):
        editor = self.query_one("#model-editor", ModelConfigEditor)
        model_name = editor.model_name
        configs = load_model_configs()
        configs[model_name] = config
        save_model_configs(configs)
        self.query_one("#model-editor-container").display = False
        self.query_one("#model-selector-container").display = True
        self.query_one("#model-selector").focus()

    async def action_personality_selected(self, personality_name: str):
        self.selected_personality = personality_name
        
        # Send personality change and clear to subprocess
        if self.runner.running:
            try:
                await self.runner.send("/clear")
                await self._read_until_prompt(timeout=5)
                await self.runner.send(f"/personality_set {personality_name}")
                await self._read_until_prompt(timeout=5)
                await asyncio.sleep(0.1)
            except Exception:
                pass
        
        # Hide personality selector and show chat/input
        self.query_one("#personality-selector-container").display = False
        self.query_one("#chat-center").display = True
        self.query_one("#input-center").display = True
        
        # Show welcome screen (Kaplumba logo) for new conversation
        await self._reset_chat_history()
        self._set_busy(False)
        self.refresh_command_menu()
        self.query_one("#input").focus()

    async def action_dismiss_personality_selector(self):
        self.query_one("#personality-selector-container").display = False
        self.query_one("#chat-center").display = True
        self.query_one("#input-center").display = True
        self.refresh_command_menu()
        self.query_one("#input").focus()

    async def show_personality_selector(self):
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#personality-selector-container").display = True
        selector = self.query_one("#personality-selector", PersonalitySelector)
        personalities = list(selector.personalities)
        names = [name for name, _ in personalities]
        selector.selected_index = names.index(self.selected_personality) if self.selected_personality in names else 0
        selector.render_list()
        selector.focus()

    async def action_submit(self):
        if self.busy or self.loading or not self.runner.running:
            return

        box = self.query_one("#input", ChatInput)
        user_text = box.text.strip()
        if not user_text:
            return

        box.clear()

        chat = self.query_one("#chat", VerticalScroll)

        # If /clear command, clear the chat display AND reset subprocess state
        if user_text == "/clear":
            await self._reset_chat_history()
            # Also tell subprocess to clear its KV cache and message history
            if self.runner.running:
                try:
                    await self.runner.send("/clear")
                    await self._read_until_prompt(timeout=5)
                except Exception:
                    pass
            self._set_busy(False)
            self.refresh_command_menu()
            self.query_one("#input").focus()
            return

        # If /models command, show model selector
        if user_text == "/models":
            await self.show_model_selector()
            return

        if user_text == "/options":
            await self.show_options_selector()
            return

        if user_text == "/personality":
            await self.show_personality_selector()
            return

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
            await self.runner.interrupt()
            self.interrupted = True

    async def action_quit(self):
        await self._stop_model_process()
        self.exit()

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
            await self._stop_model_process()
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

    async def action_reload_model(self) -> None:
        if self.runner.running:
            return
        
        # Prevent multiple simultaneous reloads
        if self.loading or self.reloading:
            return
        
        self._set_busy(False)
        self.crash_dialog_visible = False
        self.query_one("#crash-dialog-container").display = False
        
        # Clear chat history
        await self._reset_chat_history()
        self.crash_count = 0
        self.reloading = True
        self._show_loading_ui("Reloading model...")
        asyncio.create_task(self.initialize_model())

    async def on_button_pressed(self, event: Button.Pressed):
        """Handle crash dialog button presses."""
        if event.button.id == "crash-reload":
            self.crash_dialog_visible = False
            self.query_one("#crash-dialog-container").display = False
            self._show_loading_ui(f"Reloading model (crash #{self.crash_count})...")
            await self._stop_model_process()
            asyncio.create_task(self.initialize_model())
        elif event.button.id == "crash-quit":
            self.exit("Model crashed")

    async def _read_until_prompt(self, timeout=60):
        return await self.runner._read_until_prompt(timeout=timeout)

    async def run_model(self, user_text: str):
        if self.first_message:
            await asyncio.sleep(2)
            self.first_message = False

        if not self.runner.running:
            await self._handle_crash("")
            return

        user_text = " ".join(user_text.split("\n"))

        if not await self.runner.send(user_text):
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
                chunk = await asyncio.wait_for(
                    self.runner.proc.stdout.read(256), timeout=0.05
                )
            except asyncio.TimeoutError:
                if self.interrupted:
                    break
                continue
            except Exception:
                await self._handle_crash("")
                return

            if not chunk:
                await self._handle_crash("")
                return

            buf += chunk.decode(errors="ignore")

            if buf.endswith(TUI_PROMPT_MARKER):
                buf = buf[: -len(TUI_PROMPT_MARKER)]
                break

            now = asyncio.get_event_loop().time()
            if now - last_update > 0.05:
                if thinking_enabled:
                    if "</think>" not in buf:
                        spinner_index = (spinner_index + 1) % len(spinner_frames)
                        await self.current_md.update(f"Thinking... {spinner_frames[spinner_index]}")
                    else:
                        display = strip_prompt_markers(get_display_text(buf))
                        if display:
                            await self.current_md.update(f"{format_for_display(display)} ▌")
                else:
                    display = strip_prompt_markers(buf)
                    if display:
                        await self.current_md.update(f"{format_for_display(display)} ▌")
                last_update = now

        if self.interrupted:
            remaining = await self._read_until_prompt(timeout=10)
            if remaining:
                buf += remaining

        if thinking_enabled:
            display = strip_prompt_markers(get_display_text(buf))
        else:
            display = strip_prompt_markers(buf)

        if self.interrupted:
            display += "\n\n*— stopped*"
            self.interrupted = False

        try:
            await self.current_md.update(format_for_display(display))
        except Exception as e:
            # Widget might have been removed, show error in chat
            error_msg = f'<error: {e}>'
            await self.current_md.update(error_msg)
            return
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
