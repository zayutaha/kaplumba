from textual.events import Key
from textual.widgets import TextArea


class ChatInput(TextArea):
    def on_mount(self) -> None:
        self.show_line_numbers = False
        self.soft_wrap = True
        self.styles.height = 1
        self.set_interval(0.05, self.sync_height)

    def sync_height(self) -> None:
        target_height = min(max(1, self.virtual_size.height), 5)
        current = self.styles.height
        if current is None or getattr(current, 'value', current) != target_height:
            self.styles.height = target_height
            self.refresh()

    async def _on_key(self, event: Key) -> None:
        if event.key is None:
            return

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
            await self.app.action_quit()
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
