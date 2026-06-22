import json
from pathlib import Path

from textual.events import Key
from textual.widgets import Static

from model_catalog import ModelInfo, list_models
from settings_store import get_models_dir, set_models_dir


class ModelSelector(Static):
    can_focus = True
    FAVORITES_FILE = Path.home() / ".omlx" / "favorites.json"

    def __init__(self, models: list[ModelInfo], **kwargs):
        super().__init__(**kwargs)
        self.models = models
        self.selected_index = 0
        self.favorites: set[str] = self._load_favorites()
        self._editing_dir = False
        self._dir_buffer = ""
        self.render_list()

    def on_mount(self) -> None:
        self.set_interval(5, self._refresh_memory)

    def _refresh_memory(self) -> None:
        selected = self._sorted_models()[self.selected_index].name if self.models else None
        self.models = list_models({})
        if selected:
            for i, m in enumerate(self.models):
                if m.name == selected:
                    self.selected_index = i
                    break
        self.render_list()
        self.refresh()

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
            self.FAVORITES_FILE.write_text(json.dumps({"favorites": list(self.favorites)}, indent=2))
        except Exception:
            pass

    def _sorted_models(self):
        fav = [m for m in self.models if m.name in self.favorites]
        rest = [m for m in self.models if m.name not in self.favorites]
        return fav + rest

    def render_list(self):
        if self._editing_dir:
            lines = ["[bold #f0a500]Model directory[/bold #f0a500]\n"]
            lines.append("[dim]Type the new path, Enter to confirm, Esc to cancel[/dim]\n")
            lines.append(f"[bold #f0a500]🗂️[/bold #f0a500] [bold]{self._dir_buffer}[/bold]")
            self.update("\n".join(lines))
            return

        models_dir = get_models_dir()
        sorted_models = self._sorted_models()
        lines = ["[bold #f0a500]Select a model:[/bold #f0a500]\n"]
        lines.append(f"[bold #f0a500]🗂️ {models_dir}[/bold #f0a500]\n")
        for i, m in enumerate(sorted_models):
            prefix = "* " if m.name in self.favorites else "  "
            c = m.capabilities
            arch = m.model_type if m.model_type else "—"
            fit_text = f"needs {c.estimated_memory} / free {c.available_memory}"
            disabled = not c.fits_memory
            if i == self.selected_index and not disabled:
                lines.append(f"[bold #4caf50]❯ {prefix}{m.name}[/bold #4caf50]")
                lines.append(f"  [#4caf50]{m.size_gib} | {arch} | {fit_text}[/#4caf50]")
            elif i == self.selected_index and disabled:
                lines.append(f"[bold #cc6666]❯ {prefix}{m.name}[/bold #cc6666]")
                lines.append(f"  [#cc6666]{m.size_gib} | {arch} | {fit_text}[/#cc6666]")
            elif disabled:
                lines.append(f"  {prefix}{m.name}")
                lines.append(f"  [#886666]{m.size_gib} | {arch} | {fit_text}[/#886666]")
            else:
                lines.append(f"  [bold #4caf50]{prefix}{m.name}[/bold #4caf50]")
                lines.append(f"  [#4caf50]{m.size_gib} | {arch} | {fit_text}[/#4caf50]")
        lines.append("\n[dim](↑/↓ navigate, Enter select, f favorite, e edit config, d change dir, red entries are risky, Esc back, Ctrl+C quit)[/dim]")
        self.update("\n".join(lines))

    async def on_key(self, event: Key) -> None:
        if self._editing_dir:
            if event.key == "enter":
                event.prevent_default()
                if self._dir_buffer.strip():
                    set_models_dir(self._dir_buffer.strip())
                self._editing_dir = False
                self.models = list_models({})
                self.selected_index = 0
                self.render_list()
            elif event.key == "escape":
                event.prevent_default()
                self._editing_dir = False
                self.render_list()
            elif event.key == "backspace":
                event.prevent_default()
                self._dir_buffer = self._dir_buffer[:-1]
                self.render_list()
            elif event.key == "ctrl+c":
                event.prevent_default()
                self.app.exit()
            elif event.key and len(event.key) == 1:
                event.prevent_default()
                self._dir_buffer += event.key
                self.render_list()
            return

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
            await self.app.action_model_selected(self._sorted_models()[self.selected_index].name)
        elif event.key == "f":
            event.prevent_default()
            model_name = self._sorted_models()[self.selected_index].name
            self.favorites.add(model_name) if model_name not in self.favorites else self.favorites.discard(model_name)
            self._save_favorites()
            self.render_list()
        elif event.key == "e":
            event.prevent_default()
            await self.app.action_model_edit(self._sorted_models()[self.selected_index].name)
        elif event.key == "d":
            event.prevent_default()
            self._editing_dir = True
            self._dir_buffer = str(get_models_dir())
            self.render_list()
        elif event.key == "escape":
            event.prevent_default()
            await self.app.action_dismiss_model_selector()
        elif event.key == "ctrl+c":
            event.prevent_default()
            self.app.exit()
