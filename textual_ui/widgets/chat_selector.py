import datetime as dt
from pathlib import Path
from textual.events import Key
from textual.widgets import Static

class ChatSelector(Static):
    can_focus = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chats = []
        self.selected_index = 0
        self.refresh_chats()

    def refresh_chats(self):
        chats_dir = Path("chats")
        chats_dir.mkdir(exist_ok=True)
        chat_files = list(chats_dir.glob("*.json"))
        # Sort by modification time (most recent first)
        chat_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        self.chats = []
        for cf in chat_files:
            mtime = dt.datetime.fromtimestamp(cf.stat().st_mtime)
            self.chats.append({
                "name": cf.stem,
                "time": mtime.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        if self.chats:
            self.selected_index = min(self.selected_index, len(self.chats) - 1)
        else:
            self.selected_index = 0
        self.render_list()

    def render_list(self):
        lines = ["[bold #f0a500]Select a saved chat:[/bold #f0a500]\n"]
        if not self.chats:
            lines.append("  [dim]No saved chats found.[/dim]")
        else:
            for i, chat in enumerate(self.chats):
                if i == self.selected_index:
                    lines.append(f"[bold #f0a500]❯ {chat['name']}[/bold #f0a500]")
                    lines.append(f"  [dim]Last accessed: {chat['time']}[/dim]")
                else:
                    lines.append(f"  {chat['name']}")
                    lines.append(f"  [dim]Last accessed: {chat['time']}[/dim]")
        
        lines.append("\n[dim](↑/↓ navigate, Enter select, Esc back, Ctrl+C quit)[/dim]")
        self.update("\n".join(lines))

    async def on_key(self, event: Key) -> None:
        if event.key == "up":
            if self.chats:
                event.prevent_default()
                self.selected_index = (self.selected_index - 1) % len(self.chats)
                self.render_list()
        elif event.key == "down":
            if self.chats:
                event.prevent_default()
                self.selected_index = (self.selected_index + 1) % len(self.chats)
                self.render_list()
        elif event.key == "enter":
            if self.chats:
                event.prevent_default()
                chat_name = self.chats[self.selected_index]["name"]
                await self.app.action_chat_selected(chat_name)
        elif event.key == "escape":
            event.prevent_default()
            await self.app.action_dismiss_chat_selector()
        elif event.key == "ctrl+c":
            event.prevent_default()
            self.app.exit()
