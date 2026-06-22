from __future__ import annotations

import asyncio
import json
from pathlib import Path

from conversation_engine import run_model_stream
from model_interface import FakeModelPort, MLXSubprocessAdapter, ModelPort
from settings_store import (
    DEFAULT_MODEL_OPTIONS,
    get_model_options,
    get_model_personality,
    load_model_configs,
    save_model_config,
    save_model_configs,
)
from textual_ui.personas import PERSONALITIES


class Orchestrator:
    def __init__(self, chat_ui, port: ModelPort | None = None):
        self.chat = chat_ui
        self.port: ModelPort = port or MLXSubprocessAdapter()
        self.selected_model: str | None = None
        self.selected_personality = "default"
        self.model_options = dict(DEFAULT_MODEL_OPTIONS)
        self.crash_count = 0
        self.max_crashes = 3
        self.reloading = False
        self._stream_task = None

    @property
    def current_system_prompt(self) -> str:
        return PERSONALITIES.get(self.selected_personality, PERSONALITIES["default"])

    def get_model_config(self, model_name: str) -> dict:
        return load_model_configs().get(model_name, {})

    async def handle_submit(self) -> None:
        if self.chat.busy or self.chat.loading or not self.port.running:
            return

        box = self.chat.query_one("#input")
        user_text = box.text.strip()
        if not user_text:
            return
        box.clear()

        if user_text == "/clear":
            await self.chat.reset_chat()
            if self.port.running:
                try:
                    await self.port.send_command("/clear")
                except Exception:
                    pass
            self.chat._set_busy(False)
            self.chat.refresh_command_menu()
            self.chat.query_one("#input").focus()
            return

        if user_text == "/chat":
            await self.chat.show_chat_selector()
            return

        if user_text.startswith("/chat "):
            self.chat._set_busy(True)
            try:
                resp = await self.port.send_command(user_text)
                if resp:
                    # Look for JSON: history marker
                    json_start = resp.find("JSON:")
                    if json_start >= 0:
                        history_json = resp[json_start + 5 :].strip()
                        self.chat.notify(f"Received history: {len(history_json)} bytes")
                        history = json.loads(history_json)
                        await self.chat.clear_chat()
                        for msg in history:
                            role = msg["role"]
                            content = msg["content"]
                            if role == "user":
                                await self.chat.append_user_message(content)
                            elif role == "assistant":
                                await self.chat.append_assistant_message(content)
                    
                    # Also show any [INFO] messages (like the list of chats)
                    lines = [l for l in resp.splitlines() if "[INFO]" in l or "[ERROR]" in l or l.strip().startswith("- ")]
                    if lines:
                        # Convert info/list lines to a readable display
                        display = "\n".join(lines).strip()
                        if "Available chats:" in display or "not found" in display:
                            await self.chat.show_overlay("Chats", display)
                        else:
                            self.chat.notify(display)
            except Exception as e:
                self.chat.notify(f"Failed to load chat: {str(e)}", severity="error")
            
            self.chat._set_busy(False)
            self.chat.refresh_command_menu()
            self.chat.query_one("#input").focus()
            return

        if user_text == "/save" or user_text.startswith("/save "):
            self.chat._set_busy(True)
            try:
                resp = await self.port.send_command(user_text)
                if resp:
                    lines = [l for l in resp.splitlines() if "[INFO]" in l or "[ERROR]" in l]
                    if lines:
                        self.chat.notify("\n".join(lines).strip())
            except Exception:
                pass
            self.chat._set_busy(False)
            self.chat.refresh_command_menu()
            self.chat.query_one("#input").focus()
            return

        if user_text == "/models":
            await self.chat.show_model_selector()
            return
        if user_text == "/options":
            await self.chat.show_options_selector()
            return
        if user_text == "/personality":
            await self.chat.show_personality_selector()
            return

        # Non-generation slash commands handled locally
        if user_text == "/memory":
            try:
                resp = await self.port.send_command("/memory")
                display = resp or "No memory info"
                lines = [l for l in display.splitlines() if not l.startswith("[INFO]")]
                display = "\n".join(lines).strip() or display.strip()
            except Exception:
                display = "Failed to get memory info"
            await self.chat.show_overlay("Memory", display)
            self.chat.refresh_command_menu()
            self.chat.query_one("#input").focus()
            return

        if user_text == "/restore":
            self.chat._set_busy(True)
            try:
                resp = await self.port.send_command("/restore")
            except Exception:
                pass
            await self.chat.handle_stream_text(user_text)
            await self.chat.handle_stream_finished("")
            self.chat._set_busy(False)
            self.chat.refresh_command_menu()
            self.chat.query_one("#input").focus()
            return

        if user_text == "/mtp":
            try:
                resp = await self.port.send_command("/mtp")
                if resp:
                    for line in resp.splitlines():
                        line = line.strip()
                        if line.startswith("[INFO]"):
                            await self.chat.show_banner(line.replace("[INFO]", "").strip())
                            break
                    else:
                        await self.chat.show_banner("MTP toggled")
            except Exception:
                await self.chat.show_banner("Failed to toggle MTP")
            self.chat.refresh_command_menu()
            self.chat.query_one("#input").focus()
            return

        if user_text.startswith("/unload "):
            self.chat._set_busy(True)
            try:
                resp = await self.port.send_command(user_text)
            except Exception:
                pass
            await self.chat.handle_stream_text(user_text)
            await self.chat.handle_stream_finished("")
            self.chat._set_busy(False)
            self.chat.refresh_command_menu()
            self.chat.query_one("#input").focus()
            return

        await self.chat.handle_stream_text(user_text)
        self.chat._set_busy(True)

        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()

        self._stream_task = asyncio.create_task(self._run_stream(user_text))

    async def _run_stream(self, user_text: str) -> None:
        try:
            await run_model_stream(self.chat, self.port, user_text)
        except asyncio.CancelledError:
            pass

    async def handle_interrupt(self) -> None:
        if self.chat.busy and self.port:
            await self.port.interrupt()
            self.chat.interrupted = True

    async def handle_reload(self) -> None:
        if self.port.running or self.chat.loading or self.reloading:
            return
        self.chat._set_busy(False)
        self.chat.crash_dialog_visible = False
        self.chat.query_one("#crash-dialog-container").display = False
        await self.chat.reset_chat()
        self.crash_count = 0
        self.reloading = True
        self.chat.show_model_loading("Reloading model...")
        asyncio.create_task(self._load_model())

    async def handle_quit(self) -> None:
        await self.port.stop()
        self.chat.exit()

    async def handle_chat_selected(self, chat_name: str) -> None:
        self.chat._set_busy(True)
        self.chat.show_chat_ui()
        try:
            resp = await self.port.send_command(f"/chat {chat_name}")
            if resp:
                json_start = resp.find("JSON:")
                if json_start >= 0:
                    history_json = resp[json_start + 5 :].strip()
                    history = json.loads(history_json)
                    await self.chat.clear_chat()
                    for msg in history:
                        role = msg["role"]
                        content = msg["content"]
                        if role == "user":
                            await self.chat.append_user_message(content)
                        elif role == "assistant":
                            await self.chat.append_assistant_message(content)
                
                lines = [l for l in resp.splitlines() if "[INFO]" in l or "[ERROR]" in l]
                if lines:
                    self.chat.notify("\n".join(lines).strip())
        except Exception as e:
            self.chat.notify(f"Failed to load chat: {str(e)}", severity="error")
        
        self.chat._set_busy(False)
        self.chat.refresh_command_menu()
        self.chat.query_one("#input").focus()

    async def handle_model_selected(self, model_name: str) -> None:
        self.selected_model = model_name
        self.model_options = get_model_options(model_name)
        self.selected_personality = get_model_personality(model_name)
        self.chat.show_model_loading(f"Loading {model_name}...")
        await self.port.stop()
        await self._load_model()

    async def _load_model(self) -> None:
        model_path = str(Path.home() / ".omlx" / "models" / (self.selected_model or ""))
        ok = await self.port.start(model_path, self.model_options, self.current_system_prompt)
        self.reloading = False
        if ok:
            self.crash_count = 0
            await self.chat.clear_chat()
            self.chat.show_chat_ui()
        else:
            await self.handle_crash_from_chat("Model failed to initialize")

    async def handle_personality_selected(self, personality: str) -> None:
        self.selected_personality = personality
        if self.selected_model:
            configs = load_model_configs()
            model_cfg = configs.get(self.selected_model, {})
            model_cfg["personality"] = personality
            configs[self.selected_model] = model_cfg
            save_model_configs(configs)

        if self.port.running:
            await self.port.send_command(f"/personality_set {personality}")
            await self.chat.clear_chat()
            self.chat.show_chat_ui()
            return

        self.chat.show_model_selector()

    async def handle_options_selected(self, options: dict) -> None:
        if not self.selected_model:
            return
        personality = get_model_personality(self.selected_model)
        save_model_config(self.selected_model, options, personality)
        self.model_options = get_model_options(self.selected_model)
        if self.port.running:
            self.chat.show_model_loading("Reloading model...")
            await self.port.stop()
            await self._load_model()
            return
        self.chat.show_model_selector()

    async def handle_model_config_saved(self, model_name: str, config: dict) -> None:
        options = config.get("options", {})
        personality = config.get("personality", "default")
        save_model_config(model_name, options, personality)
        if self.selected_model == model_name:
            self.model_options = get_model_options(model_name)
            self.selected_personality = personality

    async def handle_crash_from_chat(self, message: str = "") -> None:
        self.chat._set_busy(False)
        self.chat.loading = True
        self.crash_count += 1

        if self.crash_count >= self.max_crashes:
            self.chat.exit("Too many crashes, giving up")
            return

        if not self.chat.query_one("#chat-center").display:
            self.reloading = True
            self.chat.show_model_loading(f"Reloading model (crash #{self.crash_count})...")
            await self.port.stop()
            asyncio.create_task(self._load_model())
            return

        self.reloading = True
        self.chat.crash_dialog_visible = True
        self.chat.query_one("#crash-dialog-container").display = True
        self.chat.query_one("#crash-message").update(
            f"Model crashed (attempt {self.crash_count}/{self.max_crashes}). Reload or quit?"
        )
        self.chat.query_one("#crash-reload").focus()
