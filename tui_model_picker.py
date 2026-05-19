import json
import re
import subprocess
from pathlib import Path

from textual.events import Key
from textual.widgets import Static

GIB = 1024 ** 3
MEMORY_SAFETY_MARGIN_BYTES = int(1.5 * GIB)


def format_bytes_gib(num_bytes: int) -> str:
    gb = num_bytes / GIB
    return f"{gb:.1f} GB"


def get_model_size_bytes(model_name: str) -> int:
    model_dir = Path.home() / ".omlx" / "models" / model_name
    if not model_dir.exists():
        return 0
    total_bytes = 0
    for file in model_dir.glob("**/*"):
        if file.is_file():
            total_bytes += file.stat().st_size
    return total_bytes


def get_model_size(model_name: str) -> str:
    return format_bytes_gib(get_model_size_bytes(model_name))


def get_available_memory_bytes() -> int | None:
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
    try:
        output = subprocess.check_output(["hostinfo"], text=True)
    except Exception:
        return None
    match = re.search(r"Primary memory available:\s+([0-9.]+)\s+gigabytes", output)
    if not match:
        return None
    return int(float(match.group(1)) * GIB)


def estimate_model_memory_bytes(model_size_bytes: int, options: dict[str, object]) -> int:
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


def get_model_capabilities(model_name: str) -> dict[str, bool]:
    model_dir = Path.home() / ".omlx" / "models" / model_name
    has_vision = False
    has_mtp = False
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                if "image_token_id" in config or "vision_config" in config:
                    has_vision = True
        except Exception:
            pass
    if (model_dir / "preprocessor_config.json").exists():
        has_mtp = True
    if (model_dir / "video_preprocessor_config.json").exists():
        has_mtp = True
    return {"vision": has_vision, "mtp": has_mtp}


def get_available_models(options: dict[str, object]) -> list[tuple[str, str, dict]]:
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


class ModelSelector(Static):
    can_focus = True
    FAVORITES_FILE = Path.home() / ".omlx" / "favorites.json"

    def __init__(self, models: list[tuple[str, str, dict]], **kwargs):
        super().__init__(**kwargs)
        self.models = models
        self.selected_index = 0
        self.favorites: set[str] = self._load_favorites()
        self.render_list()

    def _load_favorites(self) -> set[str]:
        try:
            if self.FAVORITES_FILE.exists():
                data = json.loads(self.FAVORITES_FILE.read_text())
                return set(data.get("favorites", []))
        except Exception:
            pass
        return set()

    def _save_favorites(self):
        try:
            self.FAVORITES_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.FAVORITES_FILE.write_text(
                json.dumps({"favorites": list(self.favorites)}, indent=2)
            )
        except Exception:
            pass

    def _sorted_models(self):
        fav = [m for m in self.models if m[0] in self.favorites]
        rest = [m for m in self.models if m[0] not in self.favorites]
        return fav + rest

    def render_list(self):
        sorted_models = self._sorted_models()
        lines = ["[bold #f0a500]Select a model:[/bold #f0a500]\n"]
        for i, (model_name, size, caps) in enumerate(sorted_models):
            prefix = "* " if model_name in self.favorites else "  "
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
                lines.append(f"[bold #f0a500]❯ {prefix}{model_name}[/bold #f0a500]")
                lines.append(f"  [dim]{size} | {caps_display} | {fit_text}[/dim]")
            elif i == self.selected_index and disabled:
                lines.append(f"[bold #cc6666]❯ {prefix}{model_name}[/bold #cc6666]")
                lines.append(f"  [#cc6666]{size} | {caps_display} | {fit_text}[/#cc6666]")
            elif disabled:
                lines.append(f"  {prefix}{model_name}")
                lines.append(f"  [#886666]{size} | {caps_display} | {fit_text}[/#886666]")
            else:
                lines.append(f"  {prefix}{model_name}")
                lines.append(f"  [dim]{size} | {caps_display} | {fit_text}[/dim]")
        lines.append("\n[dim](↑/↓ navigate, Enter select, f favorite, e edit config, red entries are risky, Esc back, Ctrl+C quit)[/dim]")
        self.update("\n".join(lines))

    async def on_key(self, event: Key) -> None:
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
            selected_model = self._sorted_models()[self.selected_index][0]
            await self.app.action_model_selected(selected_model)
        elif event.key == "f":
            event.prevent_default()
            model_name = self._sorted_models()[self.selected_index][0]
            if model_name in self.favorites:
                self.favorites.discard(model_name)
            else:
                self.favorites.add(model_name)
            self._save_favorites()
            self.render_list()
        elif event.key == "e":
            event.prevent_default()
            model_name = self._sorted_models()[self.selected_index][0]
            await self.app.action_model_edit(model_name)
        elif event.key == "escape":
            event.prevent_default()
            await self.app.action_dismiss_model_selector()
        elif event.key == "ctrl+c":
            event.prevent_default()
            self.app.exit()
