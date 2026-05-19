LOGO = """
██╗  ██╗ █████╗ ██████╗ ██╗     ██╗   ██╗███╗   ███╗██████╗  █████╗ 
██║ ██╔╝██╔══██╗██╔══██╗██║     ██║   ██║████╗ ████║██╔══██╗██╔══██╗
█████╔╝ ███████║██████╔╝██║     ██║   ██║██╔████╔██║██████╔╝███████║
██╔═██╗ ██╔══██║██╔═══╝ ██║     ██║   ██║██║╚██╔╝██║██╔══██╗██╔══██║
██║  ██╗██║  ██║██║     ███████╗╚██████╔╝██║ ╚═╝ ██║██████╔╝██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝
"""

WELCOME_MESSAGES = [LOGO]

CHAT_CSS = """
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
    width: 60;
    height: auto;
    max-height: 30;
    background: #1a1a1a;
    border: round #f0a500;
    padding: 2;
    color: #d8d8d8;
}

#model-editor-container {
    layout: vertical;
    width: 100%;
    height: 100%;
    align: center middle;
    display: none;
}

#model-editor {
    width: 70;
    height: auto;
    max-height: 35;
    background: #1a1a1a;
    border: round #f0a500;
    padding: 2;
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
