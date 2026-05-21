from textual.events import Key
from textual.widgets import Static

from settings_store import OPTION_SPECS, PERSONALITY_CHOICES


class OptionsSelector(Static):
    can_focus = True

    def __init__(
        self,
        options: dict[str, object],
        personality: str | None = None,
        personality_choices: list[str] | None = None,
        model_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.options = dict(options)
        self.personality = personality
        self.personality_choices = personality_choices or PERSONALITY_CHOICES
        self.model_name = model_name
        self.selected_index = 0
        self._editing = False
        self._edit_buffer = ""
        self._is_editor = model_name is not None
        self.render_list()

    def _item_count(self) -> int:
        return len(OPTION_SPECS) + (1 if self._is_editor else 0)

    def _item_spec(self, index: int) -> dict:
        if self._is_editor and index == len(OPTION_SPECS):
            return {
                "key": "personality",
                "label": "Personality",
                "choices": self.personality_choices,
                "description": "Personality applied when this model is loaded.",
            }
        return OPTION_SPECS[index]

    def _item_value(self, spec: dict) -> object:
        if spec["key"] == "personality":
            return self.personality
        return self.options.get(spec["key"])

    def set_options(self, options: dict[str, object]) -> None:
        self.options = dict(options)
        self._is_editor = False
        self.personality = None
        self.model_name = None
        self.selected_index = min(self.selected_index, self._item_count() - 1)
        self.render_list()

    def _format_value(self, value: object) -> str:
        if value is None:
            return "Auto"
        if value is True:
            return "On"
        if value is False:
            return "Off"
        return str(value)

    def _parse_value(self, raw: str, spec: dict) -> object | None:
        raw = raw.strip()
        if not raw or raw.lower() in ("auto", "none"):
            return None
        choices = spec["choices"]
        non_none = [c for c in choices if c is not None]
        sample = non_none[0] if non_none else str
        try:
            if isinstance(sample, bool):
                return raw.lower() in ("true", "on", "1", "yes")
            if isinstance(sample, int):
                return int(raw)
            if isinstance(sample, float):
                return float(raw)
            return raw
        except (ValueError, TypeError):
            return None

    def render_content(self) -> str:
        lines = []
        for index in range(self._item_count()):
            spec = self._item_spec(index)
            label = spec["label"]
            description = spec["description"]
            if self._editing and index == self.selected_index:
                val_display = f"[reverse]{self._edit_buffer} [/reverse]"
            else:
                val_display = self._format_value(self._item_value(spec))
            if index == self.selected_index:
                lines.append(f"[bold #f0a500]❯ {label}: {val_display}[/bold #f0a500]")
            else:
                lines.append(f"  {label}: {val_display}")
            lines.append(f"  [dim]{description}[/dim]")
        if self._editing:
            lines.append("\n[dim](Enter confirm, Esc cancel, Backspace delete)[/dim]")
        elif self._is_editor:
            lines.append("\n[dim](↑/↓ navigate, ←/→ change, [bold]e[/bold] edit, Esc save & back)[/dim]")
        else:
            lines.append("\n[dim](↑/↓ navigate, ←/→ change, [bold]e[/bold] edit, Enter apply, Esc back)[/dim]")
        return "\n".join(lines)

    def render_list(self) -> None:
        parts = []
        if self._is_editor:
            parts.append(f"[bold #f0a500]Config: {self.model_name}[/bold #f0a500]")
        else:
            parts.append("[bold #f0a500]Launch options:[/bold #f0a500]")
        parts.append("")
        parts.append(self.render_content())
        self.update("\n".join(parts))

    def change_value(self, direction: int) -> None:
        spec = self._item_spec(self.selected_index)
        choices = spec["choices"]
        current = self._item_value(spec)
        try:
            index = choices.index(current)
        except ValueError:
            index = 0
        index = (index + direction) % len(choices)
        if spec["key"] == "personality":
            self.personality = choices[index]
        else:
            self.options[spec["key"]] = choices[index]
        self.render_list()

    async def _commit_edit(self) -> None:
        spec = self._item_spec(self.selected_index)
        parsed = self._parse_value(self._edit_buffer, spec)
        if parsed is not None:
            if spec["key"] == "personality":
                self.personality = parsed
            else:
                self.options[spec["key"]] = parsed
        self._editing = False
        self.render_list()

    def _cancel_edit(self) -> None:
        self._editing = False
        self.render_list()

    async def on_key(self, event: Key) -> None:
        if self._editing:
            if event.key == "enter":
                event.prevent_default()
                await self._commit_edit()
            elif event.key == "escape":
                event.prevent_default()
                self._cancel_edit()
            elif event.key == "backspace":
                event.prevent_default()
                self._edit_buffer = self._edit_buffer[:-1]
                self.render_list()
            elif event.key == "full_stop":
                event.prevent_default()
                self._edit_buffer += "."
                self.render_list()
            elif event.key == "hyphen" or event.key == "minus":
                event.prevent_default()
                self._edit_buffer += "-"
                self.render_list()
            elif event.key not in (
                "up", "down", "left", "right", "enter", "escape",
                "backspace", "tab", "ctrl+c", "ctrl+d", "ctrl+z",
                "full_stop", "hyphen", "minus",
            ):
                event.prevent_default()
                self._edit_buffer += event.key
                self.render_list()
            return

        if event.key == "up":
            event.prevent_default()
            self.selected_index = (self.selected_index - 1) % self._item_count()
            self.render_list()
        elif event.key == "down":
            event.prevent_default()
            self.selected_index = (self.selected_index + 1) % self._item_count()
            self.render_list()
        elif event.key == "left":
            event.prevent_default()
            self.change_value(-1)
        elif event.key == "right":
            event.prevent_default()
            self.change_value(1)
        elif event.key == "enter":
            event.prevent_default()
            if not self._is_editor:
                await self.app.action_options_selected(dict(self.options))
        elif event.key == "escape":
            event.prevent_default()
            if self._is_editor:
                await self._save_editor_config()
            else:
                await self.app.action_dismiss_options_selector()
        elif event.key == "ctrl+c":
            event.prevent_default()
            self.app.exit()
        elif event.key == "e":
            event.prevent_default()
            self._editing = True
            self._edit_buffer = ""
            self.render_list()

    async def _save_editor_config(self) -> None:
        opts = dict(self.options)
        personality = self.personality or "default"
        config = {"options": opts, "personality": personality}
        await self.app.action_model_editor_save(self.model_name or "", config)

    def set_editor_mode(
        self, model_name: str, options: dict, personality: str | None = None
    ) -> None:
        self.model_name = model_name
        self.options = dict(options)
        self.personality = personality
        self.personality_choices = PERSONALITY_CHOICES
        self._is_editor = True
        self.selected_index = 0
        self._editing = False
        self.render_list()
