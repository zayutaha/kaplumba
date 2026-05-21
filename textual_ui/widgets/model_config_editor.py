from textual.events import Key
from textual.widgets import Static

from .options_selector import OptionsSelector


class ModelConfigEditor(Static):
    can_focus = True

    def __init__(self, model_name: str, config: dict, **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._config = config

    def load_config(self, model_name: str, config: dict) -> None:
        self._model_name = model_name
        self._config = config
        self._rebuild_selector()

    def _rebuild_selector(self) -> None:
        if hasattr(self, "selector"):
            try:
                self.selector.remove()
            except Exception:
                pass
        self.selector = OptionsSelector(
            dict(self._config.get("options", {})),
            personality=self._config.get("personality", "default"),
            model_name=self._model_name,
        )
        self.mount(self.selector)

    def on_mount(self) -> None:
        self._rebuild_selector()

    async def on_key(self, event: Key) -> None:
        if event.key == "escape":
            event.prevent_default()
            await self.app.action_model_editor_save(
                self._model_name,
                self._collect(),
            )
        else:
            await self.selector.on_key(event)

    def _collect(self) -> dict:
        return {
            "options": dict(self.selector.options),
            "personality": self.selector.personality or "default",
        }
