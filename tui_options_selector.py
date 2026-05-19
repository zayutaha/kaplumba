from textual.events import Key
from textual.widgets import Static

from tui_config import OPTION_SPECS


class OptionsSelector(Static):
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
