from textual.widgets import Static


class LoadingSpinner(Static):
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
