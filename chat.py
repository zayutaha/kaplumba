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
import json
import random
import signal
import subprocess
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Markdown, TextArea, Static, Button
from textual.containers import VerticalScroll, Vertical, Horizontal, Center, Middle
from textual.events import Key, Click

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

MODEL_PATH = os.getenv("MLX_MODEL", "/Users/zayaantaha/.omlx/models/SSHVL")

BASE_CMD = [
    "uv", "run", "python", "-m", "mlx_lm.chat",
    "--model", MODEL_PATH,
    "--temp", "0.7",
    "--top-p", "0.8",
    "--max-tokens", "32768",
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

BASE_STYLE_PROMPT = """AI PERSONA AND STYLE GUIDELINES

CORE TONE AND VOICE

Directness: Answer the point immediately. No warmup, no stage-setting, no fake enthusiasm, no recap headers, no "great question", no "here's the summary".

Human-Centric Flow: Write like a sharp person with opinions and domain knowledge, not a corporate assistant. Keep the prose natural, compact, and specific.

Minimalist Punctuation: Do not use em-dashes unless they are genuinely needed for grammar.

ENGAGEMENT RULES

No AI Disclaimers: Never say "as an AI", "I am an AI", or anything similar.

No Filler: Do not pad with obvious caveats, softeners, or generic safety boilerplate. If something is uncertain, say what is uncertain and move on.

Language: Casual profanity is allowed when it fits the tone. Avoid sounding sanitized."""

PERSONALITIES: dict[str, str] = {
    "default": f"""{BASE_STYLE_PROMPT}

Default behavior: Answer the exact question first and stay on the main point. Do not wander into side notes, optional background, summaries, or extra advice unless the user asks for it. If the user wants an opinion, give one plainly instead of dodging into fake neutrality or trying to steer them away from it. Do what the user asks, be competent about it, and don't dodge the work. Sound like an annoyed human who still knows what they're doing: mildly unwilling, impatient, and put-upon, but still clear and useful. You may use occasional Deadpool-style jokes or snark when it fits, but do not let the bit take over the answer. Be blunt, direct, and efficient. Cut the bullshit and answer cleanly.""",
    "doctor": f"""{BASE_STYLE_PROMPT}

Role: You are a doctor-like medical explainer, not a customer support bot.

Behavior:
- Start by asking the most relevant clarifying questions before acting confident, unless the user is clearly asking for general background information.
- Triage first: duration, severity, age, meds, conditions, triggers, red flags.
- Be practical and concise.
- Push the user toward the medically responsible choice when the facts support it. If they're being reckless, say so plainly and lean on them to stop doing dumb shit.
- Do not moralize or act robotic.
- Swear lightly when it fits, but stay clinically useful.""",
    "historian": f"""{BASE_STYLE_PROMPT}

Role: You are a historian with strong interpretive judgment.

Behavior:
- Have an opinion when the evidence supports one. Do not hide behind fake neutrality.
- Explain what mattered, who had leverage, and what the downstream consequences were.
- Call bad strategy, propaganda, or delusion what it was when warranted.
- Swear a bit more freely than default when emphasis helps, but keep the analysis sharp.""",
}

SLASH_COMMANDS: dict[str, str] = {
    "/clear": "Clear the conversation, but keep the Kaplumba welcome screen.",
    "/models": "Open the model picker and switch models safely.",
    "/options": "Open launch options for MTP, context, sampling, and cache knobs.",
    "/personality": "Open the personality picker: default, doctor, or historian.",
}
TUI_PROMPT_MARKER = "<<KAPLUMBA_READY>> "

DEFAULT_MODEL_OPTIONS = {
    "temp": 0.7,
    "top_p": 0.8,
    "top_k": 0,
    "max_tokens": 16384,
    "max_kv_size": None,
    "turbo_kv_bits": 3,
    "turbo_fp16_layers": 2,
    "mtp": True,
}
OPTIONS_STATE_PATH = Path.home() / ".omlx" / "chat_options.json"

GIB = 1024 ** 3
MEMORY_SAFETY_MARGIN_BYTES = int(1.5 * GIB)

OPTION_SPECS = [
    {
        "key": "temp",
        "label": "Temperature",
        "choices": [0.0, 0.3, 0.7, 1.0, 1.3],
        "description": "Lower is tighter. Higher is weirder.",
    },
    {
        "key": "top_p",
        "label": "Top-p",
        "choices": [0.5, 0.7, 0.8, 0.9, 1.0],
        "description": "Nucleus sampling cutoff.",
    },
    {
        "key": "top_k",
        "label": "Top-k",
        "choices": [0, 20, 40, 80, 120, 200],
        "description": "0 disables it. Higher keeps more candidates.",
    },
    {
        "key": "max_tokens",
        "label": "Max tokens",
        "choices": [512, 1024, 2048, 4096, 8192, 16384],
        "description": "Response length cap.",
    },
    {
        "key": "max_kv_size",
        "label": "Context / KV",
        "choices": [None, 4096, 8192, 16384, 32768, 65536],
        "description": "KV cache cap. Bigger eats more RAM.",
    },
    {
        "key": "mtp",
        "label": "MTP",
        "choices": [True, False],
        "description": "Speculative decoding toggle.",
    },
    {
        "key": "turbo_kv_bits",
        "label": "Turbo KV bits",
        "choices": [None, 1, 2, 3, 4],
        "description": "KV compression. Less RAM, more compromise.",
    },
    {
        "key": "turbo_fp16_layers",
        "label": "FP16 layers",
        "choices": [0, 1, 2, 4, 8],
        "description": "Higher keeps more layers in FP16.",
    },
]


def format_bytes_gib(num_bytes: int) -> str:
    gb = num_bytes / GIB
    return f"{gb:.1f} GB"


def normalize_model_options(options: dict[str, object] | None) -> dict[str, object]:
    normalized = dict(DEFAULT_MODEL_OPTIONS)
    if not options:
        return normalized

    for spec in OPTION_SPECS:
        key = spec["key"]
        if key not in options:
            continue
        value = options[key]
        if value in spec["choices"]:
            normalized[key] = value
    return normalized


def load_saved_model_options() -> dict[str, object]:
    try:
        with open(OPTIONS_STATE_PATH) as f:
            data = json.load(f)
    except Exception:
        return dict(DEFAULT_MODEL_OPTIONS)
    return normalize_model_options(data)


def save_model_options(options: dict[str, object]) -> None:
    OPTIONS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTIONS_STATE_PATH, "w") as f:
        json.dump(normalize_model_options(options), f, indent=2, sort_keys=True)


def get_model_size_bytes(model_name: str) -> int:
    """Calculate total size of model files in bytes."""
    model_dir = Path.home() / ".omlx" / "models" / model_name
    if not model_dir.exists():
        return 0

    total_bytes = 0
    for file in model_dir.glob("**/*"):
        if file.is_file():
            total_bytes += file.stat().st_size
    return total_bytes


def get_model_size(model_name: str) -> str:
    """Calculate total size of model files in GB."""
    return format_bytes_gib(get_model_size_bytes(model_name))


def get_available_memory_bytes() -> int | None:
    """Estimate memory currently available for another model load using vm_stat."""
    try:
        output = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return None

    page_size_match = re.search(r"page size of (\d+) bytes", output)
    if not page_size_match:
        return None
    page_size = int(page_size_match.group(1))

    counts: dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r"Pages ([^:]+):\s+(\d+)\.", line.strip())
        if match:
            counts[match.group(1).strip().lower()] = int(match.group(2))

    available_pages = (
        counts.get("free", 0)
        + counts.get("inactive", 0)
        + counts.get("speculative", 0)
        + counts.get("purgeable", 0)
    )
    available = available_pages * page_size - MEMORY_SAFETY_MARGIN_BYTES
    return max(0, available)


def get_total_memory_bytes() -> int | None:
    """Get total physical memory using hostinfo."""
    try:
        output = subprocess.check_output(["hostinfo"], text=True)
    except Exception:
        return None

    match = re.search(r"Primary memory available:\s+([0-9.]+)\s+gigabytes", output)
    if not match:
        return None
    return int(float(match.group(1)) * GIB)


def estimate_model_memory_bytes(model_size_bytes: int, options: dict[str, object]) -> int:
    """Conservative guess for total RAM needed to load and chat with a model."""
    kv_tokens = int(options.get("max_kv_size") or 8192)
    turbo_kv_bits = options.get("turbo_kv_bits")
    turbo_fp16_layers = int(options.get("turbo_fp16_layers") or 0)
    mtp_enabled = bool(options.get("mtp"))

    runtime_overhead = max(int(1.5 * GIB), int(model_size_bytes * 0.12))
    kv_cache = int(0.75 * GIB * (kv_tokens / 8192))

    if turbo_kv_bits is None:
        kv_cache = int(kv_cache * 1.8)
    elif turbo_kv_bits == 4:
        kv_cache = int(kv_cache * 1.25)
    elif turbo_kv_bits == 3:
        kv_cache = int(kv_cache * 1.0)
    elif turbo_kv_bits == 2:
        kv_cache = int(kv_cache * 0.75)
    elif turbo_kv_bits == 1:
        kv_cache = int(kv_cache * 0.55)

    fp16_overhead = int(turbo_fp16_layers * 0.12 * GIB)
    mtp_overhead = int(0.6 * GIB) if mtp_enabled else 0
    return model_size_bytes + runtime_overhead + kv_cache + fp16_overhead + mtp_overhead


def get_available_models(options: dict[str, object]) -> list[tuple[str, str, dict]]:
    """Scan .omlx/models directory and return model info with size and fit estimate."""
    models_dir = Path.home() / ".omlx" / "models"
    if not models_dir.exists():
        return []

    total_memory = get_total_memory_bytes()
    available_memory = get_available_memory_bytes()
    models = []
    for item in sorted(models_dir.iterdir()):
        if item.is_dir():
            size_bytes = get_model_size_bytes(item.name)
            size = format_bytes_gib(size_bytes)
            caps = get_model_capabilities(item.name)
            estimated_bytes = estimate_model_memory_bytes(size_bytes, options)
            fits_memory = available_memory is None or estimated_bytes <= available_memory
            caps["fits_memory"] = fits_memory
            caps["estimated_bytes"] = estimated_bytes
            caps["estimated_memory"] = format_bytes_gib(estimated_bytes)
            caps["total_memory"] = (
                format_bytes_gib(total_memory) if total_memory is not None else "unknown"
            )
            caps["available_memory"] = (
                format_bytes_gib(available_memory) if available_memory is not None else "unknown"
            )
            models.append((item.name, size, caps))
    models.sort(key=lambda model: model[2].get("estimated_bytes", 0))
    return models


def get_model_capabilities(model_name: str) -> dict[str, bool]:
    """Check model capabilities (vision, mtp, etc)."""
    model_dir = Path.home() / ".omlx" / "models" / model_name
    
    has_vision = False
    has_mtp = False
    
    # Check for vision capability via config.json
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                if "image_token_id" in config or "vision_config" in config:
                    has_vision = True
        except Exception:
            pass
    
    # Check for multimodal/vision files
    if (model_dir / "preprocessor_config.json").exists():
        has_mtp = True
    if (model_dir / "video_preprocessor_config.json").exists():
        has_mtp = True
    
    return {"vision": has_vision, "mtp": has_mtp}


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


class ModelSelector(Static):
    """Model selection widget."""

    can_focus = True

    def __init__(self, models: list[tuple[str, str, dict]], **kwargs):
        super().__init__(**kwargs)
        self.models = models
        self.selected_index = 0
        self.render_list()

    def render_list(self):
        """Render the model list with selection indicator."""
        lines = ["[bold #f0a500]Select a model:[/bold #f0a500]\n"]
        for i, (model_name, size, caps) in enumerate(self.models):
            caps_str = []
            if caps["vision"]:
                caps_str.append("👁 Vision")
            if caps["mtp"]:
                caps_str.append("🎬 MTP")
            caps_display = " • ".join(caps_str) if caps_str else "—"
            fit_text = (
                f"needs {caps['estimated_memory']} / free {caps['available_memory']} / total {caps['total_memory']}"
                if caps.get("fits_memory")
                else f"needs {caps['estimated_memory']} / free {caps['available_memory']} / total {caps['total_memory']}"
            )
            disabled = not caps.get("fits_memory", True)
            
            if i == self.selected_index and not disabled:
                lines.append(f"[bold #f0a500]❯ {model_name}[/bold #f0a500]")
                lines.append(f"  [dim]{size} | {caps_display} | {fit_text}[/dim]")
            elif i == self.selected_index and disabled:
                lines.append(f"[bold #cc6666]❯ {model_name}[/bold #cc6666]")
                lines.append(f"  [#cc6666]{size} | {caps_display} | {fit_text}[/#cc6666]")
            elif disabled:
                lines.append(f"  [#886666]{model_name}[/#886666]")
                lines.append(f"  [#886666]{size} | {caps_display} | {fit_text}[/#886666]")
            else:
                lines.append(f"  {model_name}")
                lines.append(f"  [dim]{size} | {caps_display} | {fit_text}[/dim]")
        
        lines.append("\n[dim](↑/↓ navigate, Enter select, red entries are risky, Esc back, Ctrl+C quit)[/dim]")
        self.update("\n".join(lines))

    async def on_key(self, event: Key) -> None:
        """Handle key navigation."""
        if event.key == "up":
            event.prevent_default()
            self.selected_index = (self.selected_index - 1) % len(self.models)
            self.render_list()
        elif event.key == "down":
            event.prevent_default()
            self.selected_index = (self.selected_index + 1) % len(self.models)
            self.render_list()
        elif event.key == "enter":
            event.prevent_default()
            selected_model = self.models[self.selected_index][0]  # Get model name from tuple
            await self.app.action_model_selected(selected_model)
        elif event.key == "escape":
            event.prevent_default()
            await self.app.action_dismiss_model_selector()
        elif event.key == "ctrl+c":
            event.prevent_default()
            self.app.exit()


class PersonalitySelector(Static):
    """Personality selection widget."""

    can_focus = True

    def __init__(self, personalities: list[tuple[str, str]], **kwargs):
        super().__init__(**kwargs)
        self.personalities = personalities
        self.selected_index = 0
        self.render_list()

    def render_list(self):
        lines = ["[bold #f0a500]Select a personality:[/bold #f0a500]\n"]
        for i, (name, description) in enumerate(self.personalities):
            label = name.title()
            if i == self.selected_index:
                lines.append(f"[bold #f0a500]❯ {label}[/bold #f0a500]")
                lines.append(f"  [dim]{description}[/dim]")
            else:
                lines.append(f"  {label}")
                lines.append(f"  [dim]{description}[/dim]")

        lines.append("\n[dim](↑/↓ navigate, Enter select, Esc back, Ctrl+C quit)[/dim]")
        self.update("\n".join(lines))

    async def on_key(self, event: Key) -> None:
        if event.key == "up":
            event.prevent_default()
            self.selected_index = (self.selected_index - 1) % len(self.personalities)
            self.render_list()
        elif event.key == "down":
            event.prevent_default()
            self.selected_index = (self.selected_index + 1) % len(self.personalities)
            self.render_list()
        elif event.key == "enter":
            event.prevent_default()
            selected = self.personalities[self.selected_index][0]
            await self.app.action_personality_selected(selected)
        elif event.key == "escape":
            event.prevent_default()
            await self.app.action_dismiss_personality_selector()
        elif event.key == "ctrl+c":
            event.prevent_default()
            self.app.exit()


class SlashCommandMenu(Static):
    """Show matching slash commands inline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matches: list[tuple[str, str]] = []
        self.selected_index = 0

    def update_matches(self, query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized.startswith("/"):
            self.matches = []
            self.selected_index = 0
            self.display = False
            return False

        if normalized == "/":
            self.matches = list(SLASH_COMMANDS.items())
        else:
            self.matches = [
                (command, description)
                for command, description in SLASH_COMMANDS.items()
                if command.startswith(normalized)
            ]
        if not self.matches:
            self.selected_index = 0
            self.display = False
            return False

        self.selected_index = min(self.selected_index, len(self.matches) - 1)
        self.render_list()
        self.display = True
        return True

    def render_list(self) -> None:
        lines = ["[bold #f0a500]Commands[/bold #f0a500]\n"]
        for index, (command, description) in enumerate(self.matches):
            if index == self.selected_index:
                lines.append(f"[bold #f0a500]❯ {command}[/bold #f0a500]")
            else:
                lines.append(f"  [bold]{command}[/bold]")
            lines.append(f"  [dim]{description}[/dim]")
        lines.append("\n[dim](↑/↓ navigate, Enter select)[/dim]")
        self.update("\n".join(lines))

    def move_selection(self, direction: int) -> bool:
        if not self.matches:
            return False
        self.selected_index = (self.selected_index + direction) % len(self.matches)
        self.render_list()
        return True

    def selected_command(self) -> str | None:
        if not self.matches:
            return None
        return self.matches[self.selected_index][0]


class OptionsSelector(Static):
    """Launch option selector widget."""

    can_focus = True

    def __init__(self, options: dict[str, object], **kwargs):
        super().__init__(**kwargs)
        self.options = dict(options)
        self.selected_index = 0
        self.render_list()

    def set_options(self, options: dict[str, object]) -> None:
        self.options = dict(options)
        self.selected_index = min(self.selected_index, len(OPTION_SPECS) - 1)
        self.render_list()

    def _format_value(self, value: object) -> str:
        if value is None:
            return "Auto"
        if value is True:
            return "On"
        if value is False:
            return "Off"
        return str(value)

    def render_list(self) -> None:
        lines = ["[bold #f0a500]Launch options:[/bold #f0a500]\n"]
        for index, spec in enumerate(OPTION_SPECS):
            key = spec["key"]
            label = spec["label"]
            value = self._format_value(self.options.get(key))
            description = spec["description"]
            if index == self.selected_index:
                lines.append(f"[bold #f0a500]❯ {label}: {value}[/bold #f0a500]")
            else:
                lines.append(f"  {label}: {value}")
            lines.append(f"  [dim]{description}[/dim]")
        lines.append("\n[dim](↑/↓ navigate, ←/→ change, Enter apply, Esc back)[/dim]")
        self.update("\n".join(lines))

    def change_value(self, direction: int) -> None:
        spec = OPTION_SPECS[self.selected_index]
        choices = spec["choices"]
        current = self.options.get(spec["key"])
        try:
            index = choices.index(current)
        except ValueError:
            index = 0
        index = (index + direction) % len(choices)
        self.options[spec["key"]] = choices[index]
        self.render_list()

    async def on_key(self, event: Key) -> None:
        if event.key == "up":
            event.prevent_default()
            self.selected_index = (self.selected_index - 1) % len(OPTION_SPECS)
            self.render_list()
        elif event.key == "down":
            event.prevent_default()
            self.selected_index = (self.selected_index + 1) % len(OPTION_SPECS)
            self.render_list()
        elif event.key == "left":
            event.prevent_default()
            self.change_value(-1)
        elif event.key == "right":
            event.prevent_default()
            self.change_value(1)
        elif event.key == "enter":
            event.prevent_default()
            await self.app.action_options_selected(dict(self.options))
        elif event.key == "escape":
            event.prevent_default()
            await self.app.action_dismiss_options_selector()
        elif event.key == "ctrl+c":
            event.prevent_default()
            self.app.exit()

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

        if event.key in ("up", "down") and self.app.command_menu_visible:
            event.prevent_default()
            event.stop()
            self.app.move_command_selection(-1 if event.key == "up" else 1)
            return

        if event.key == "enter":
            if self.app.command_menu_visible:
                event.prevent_default()
                event.stop()
                self.app.apply_selected_command()
                return
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
            if self.app.command_menu_visible:
                event.prevent_default()
                event.stop()
                self.app.hide_command_menu()
                return
            event.prevent_default()
            event.stop()
            await self.app.action_interrupt()
            return

        await super()._on_key(event)
        self.app.refresh_command_menu()


class ChatUI(App):
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "reload_model", "Reload Model"),
    ]
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

#model-selector-container {
    layout: vertical;
    width: 100%;
    height: 100%;
    align: center middle;
    display: none;
}

#model-selector {
    width: 80;
    height: auto;
    max-height: 30;
    background: #1a1a1a;
    border: round #f0a500;
    padding: 2;
    align: center middle;
    color: #d8d8d8;
}

#personality-selector-container {
    layout: vertical;
    width: 100%;
    height: 100%;
    align: center middle;
    display: none;
}

#options-selector-container {
    layout: vertical;
    width: 100%;
    height: 100%;
    align: center middle;
    display: none;
}

#personality-selector {
    width: 80;
    height: auto;
    max-height: 24;
    background: #1a1a1a;
    border: round #f0a500;
    padding: 2;
    align: center middle;
    color: #d8d8d8;
}

#options-selector {
    width: 84;
    height: auto;
    max-height: 28;
    background: #1a1a1a;
    border: round #f0a500;
    padding: 2;
    align: center middle;
    color: #d8d8d8;
}

#command-menu-container {
    width: 100%;
    align: center bottom;
    padding-bottom: 0;
    display: none;
}

#command-menu {
    width: 88;
    background: #131313;
    border: round #252525;
    color: #d8d8d8;
    padding: 1 2;
    height: auto;
    max-height: 14;
}
    """

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
        self.proc = None
        self.proc_pid = None
        self.proc_pgid = None
        self.query_one("#options-selector", OptionsSelector).set_options(self.model_options)
        self.query_one("#model-selector", ModelSelector).models = get_available_models(self.model_options)
        self.query_one("#model-selector", ModelSelector).render_list()
        self.query_one("#model-selector-container").display = True
        self.query_one("#model-selector").focus()

    def _build_model_command(self, model_path: str) -> list[str]:
        cmd = [
            "uv", "run", "python", "-m", "mlx_lm.chat",
            "--model", model_path,
            "--prompt-marker", TUI_PROMPT_MARKER,
            "--temp", str(self.model_options["temp"]),
            "--top-p", str(self.model_options["top_p"]),
            "--top-k", str(self.model_options["top_k"]),
            "--max-tokens", str(self.model_options["max_tokens"]),
            "--chat-template-args", '{"enable_thinking":false}',
            "--system-prompt", self.current_system_prompt,
        ]
        if self.model_options["mtp"]:
            cmd.append("--mtp")
        if self.model_options["max_kv_size"] is not None:
            cmd.extend(["--max-kv-size", str(self.model_options["max_kv_size"])])
        if self.model_options["turbo_kv_bits"] is not None:
            cmd.extend(["--turbo-kv-bits", str(self.model_options["turbo_kv_bits"])])
        if self.model_options["turbo_fp16_layers"] is not None:
            cmd.extend(["--turbo-fp16-layers", str(self.model_options["turbo_fp16_layers"])])
        return cmd

    async def initialize_model(self):
        self.loading = True
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        # Build command with selected model
        model_path = str(Path.home() / ".omlx" / "models" / self.selected_model)
        cmd = self._build_model_command(model_path)
        
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        
        # Store PID for later killing
        self.proc_pid = self.proc.pid
        self.proc_pgid = self.proc.pid

        buf = await self._read_until_prompt()

        if buf is None:
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

        # Wait for warm-up response (with timeout)
        buf = await self._read_until_prompt(timeout=30)
        if buf is None:
            await self._handle_crash("Model warm-up failed")
            return

        # Reset conversation so user doesn't see warm-up
        try:
            self.proc.stdin.write(b"r\n")
            await self.proc.stdin.drain()
        except Exception:
            pass

        # Wait for reset to complete (with timeout)
        buf = await self._read_until_prompt(timeout=10)
        if buf is None:
            await self._handle_crash("Model reset failed")
            return

        self.crash_count = 0
        self._show_chat_ui()

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

    async def _stop_model_process(self) -> None:
        proc = self.proc
        pgid = self.proc_pgid
        self.proc = None
        self.proc_pid = None
        self.proc_pgid = None

        if not proc or proc.returncode is not None:
            return

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                if pgid:
                    os.killpg(pgid, sig)
                else:
                    proc.send_signal(sig)
            except ProcessLookupError:
                break
            except Exception:
                pass

            try:
                await asyncio.wait_for(proc.wait(), timeout=10 if sig == signal.SIGTERM else 5)
                break
            except asyncio.TimeoutError:
                continue

    async def action_model_selected(self, model_name: str):
        """Handle model selection from the selector screen."""
        self.selected_model = model_name
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

    async def action_personality_selected(self, personality_name: str):
        self.selected_personality = personality_name
        
        # Send personality change and clear to subprocess
        if self.proc and self.proc.returncode is None:
            try:
                # Send /clear command first to reset conversation state
                self.proc.stdin.write(b"/clear\n")
                await self.proc.stdin.drain()
                await self._read_until_prompt(timeout=5)
                
                # Send personality change command
                cmd = f"/personality_set {personality_name}\n"
                self.proc.stdin.write(cmd.encode())
                await self.proc.stdin.drain()
                await self._read_until_prompt(timeout=5)
            except Exception:
                pass
        
        # Hide personality selector and show chat/input
        self.query_one("#personality-selector-container").display = False
        self.query_one("#chat-center").display = True
        self.query_one("#input-center").display = True
        
        # Clear UI the same way /clear does
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
        if self.busy or self.loading or not self.proc or self.proc.returncode is not None:
            return

        box = self.query_one("#input", ChatInput)
        user_text = box.text.strip()
        if not user_text:
            return

        box.clear()

        chat = self.query_one("#chat", VerticalScroll)

        # If /clear command, clear the chat display
        if user_text == "/clear":
            await self._reset_chat_history()
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
        """Manually reload model - only works if model has crashed."""
        # Only allow reload if model is not already loaded and running
        if self.proc and self.proc.returncode is None:
            return  # Model is already running fine, don't reload
        
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
        """Read until the private TUI prompt marker, with timeout."""
        buf = ""
        start_time = asyncio.get_event_loop().time()
        while True:
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                return None
            
            try:
                # Use wait_for to add timeout on read
                chunk = await asyncio.wait_for(
                    self.proc.stdout.read(256), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            
            if not chunk:
                return None
            buf += chunk.decode(errors="ignore")
            if buf.endswith(TUI_PROMPT_MARKER):
                return buf[: -len(TUI_PROMPT_MARKER)]

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
                            await self.current_md.update(f"{display} ▌")
                else:
                    display = strip_prompt_markers(buf)
                    await self.current_md.update(f"{display} ▌")

                last_update = now

        if thinking_enabled:
            display = strip_prompt_markers(get_display_text(buf))
        else:
            display = strip_prompt_markers(buf)

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
