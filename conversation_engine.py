import asyncio

from textual.containers import VerticalScroll

from textual_ui.latex import format_for_display, strip_prompt_markers


THINKING_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


async def run_model_stream(chat, port, user_text: str):
    if chat.first_message:
        await asyncio.sleep(2)
        chat.first_message = False

    if not port or not port.running:
        if chat._on_crash:
            await chat._on_crash()
        return

    chat_widget = chat.query_one("#chat", VerticalScroll)
    thinking_enabled = user_text.startswith("/think")
    buf = ""
    thinking_spinner_frame = 0

    def get_display_text(buffer):
        last_end = buffer.rfind("</think>")
        if last_end >= 0:
            return buffer[last_end + len("</think>"):].strip()
        return ""

    try:
        async for chunk in port.send_message(user_text):
            buf += chunk
            if thinking_enabled:
                if "</think>" not in buf:
                    spinner_char = THINKING_SPINNER[thinking_spinner_frame % len(THINKING_SPINNER)]
                    await chat.handle_stream_chunk(f"Thinking {spinner_char}", show_cursor=False)
                    thinking_spinner_frame += 1
                else:
                    display = strip_prompt_markers(get_display_text(buf))
                    if display:
                        await chat.handle_stream_chunk(format_for_display(display))
            else:
                display = strip_prompt_markers(buf)
                if display:
                    await chat.handle_stream_chunk(format_for_display(display))
    except Exception:
        if chat._on_crash:
            await chat._on_crash()
        return

    if chat.interrupted:
        display = strip_prompt_markers(buf) + "\n\n*— stopped*"
        chat.interrupted = False
    elif thinking_enabled:
        display = strip_prompt_markers(get_display_text(buf))
    else:
        display = strip_prompt_markers(buf)

    try:
        await chat.handle_stream_finished(format_for_display(display))
    except Exception:
        pass

    scroll_y = chat_widget.scroll_offset.y
    virtual_h = chat_widget.virtual_size.height
    widget_h = chat_widget.region.height
    if virtual_h <= widget_h or (virtual_h - widget_h - scroll_y) <= 50:
        chat_widget.scroll_end(animate=False)

    chat._set_busy(False)
