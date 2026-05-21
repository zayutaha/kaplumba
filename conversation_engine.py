import asyncio
import re

from textual.containers import VerticalScroll

from textual_ui.latex import format_for_display, strip_prompt_markers


THINKING_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _detect_thinking_start(text: str) -> bool:
    """Check if text contains thinking tag opening markers."""
    return "<think>" in text or "<|channel>" in text


def _has_thinking_end(text: str) -> bool:
    """Check if text contains thinking tag closing markers."""
    return "</think>" in text or "<channel|>" in text


def _remove_thinking_blocks(text: str) -> str:
    """Remove all thinking block content (everything between opening and closing tags)."""
    # Remove Qwen thinking blocks: <think>...content...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove Gemma thinking blocks: <|channel>...content...<channel|>
    text = re.sub(r'<\|channel>.*?<channel\|>', '', text, flags=re.DOTALL)
    return text


async def run_model_stream(chat, port, user_text: str):
    if chat.first_message:
        await asyncio.sleep(2)
        chat.first_message = False

    if not port or not port.running:
        if chat._on_crash:
            await chat._on_crash()
        return

    chat_widget = chat.query_one("#chat", VerticalScroll)
    explicit_thinking = user_text.startswith("/think")
    buf = ""
    thinking_spinner_frame = 0
    in_thinking = False  # Track if we're currently in a thinking block
    thinking_processed = False  # Track if we've already handled thinking

    def get_display_text(buffer):
        """Extract text after the last closing think tag."""
        # Handle both </think> and <channel|>
        last_think_end = buffer.rfind("</think>")
        last_channel_end = buffer.rfind("<channel|>")
        
        if last_think_end > last_channel_end:
            last_end = last_think_end + len("</think>")
        elif last_channel_end >= 0:
            last_end = last_channel_end + len("<channel|>")
        else:
            return ""
            
        if last_end > 0:
            return buffer[last_end:].strip()
        return ""

    try:
        async for chunk in port.send_message(user_text):
            buf += chunk
            
            # Auto-detect thinking if it appears (for Gemma 4) and we haven't processed it yet
            if not thinking_processed and not in_thinking and _detect_thinking_start(buf):
                in_thinking = True
            
            # Handle thinking mode (either explicit /think or auto-detected)
            if in_thinking or explicit_thinking:
                if not _has_thinking_end(buf):
                    # Still thinking - show spinner
                    spinner_char = THINKING_SPINNER[thinking_spinner_frame % len(THINKING_SPINNER)]
                    await chat.handle_stream_chunk(f"Thinking {spinner_char}", show_cursor=False)
                    thinking_spinner_frame += 1
                else:
                    # Thinking is done - extract content after thinking block
                    display = strip_prompt_markers(get_display_text(buf))
                    if display:
                        await chat.handle_stream_chunk(format_for_display(display))
                    in_thinking = False
                    thinking_processed = True
            elif thinking_processed:
                # After thinking was processed, continue showing content
                display = strip_prompt_markers(get_display_text(buf))
                if display:
                    await chat.handle_stream_chunk(format_for_display(display))
            else:
                # No thinking - display normally
                display = strip_prompt_markers(buf)
                if display:
                    await chat.handle_stream_chunk(format_for_display(display))
    except Exception:
        if chat._on_crash:
            await chat._on_crash()
        return

    if chat.interrupted:
        display = _remove_thinking_blocks(strip_prompt_markers(buf)) + "\n\n*— stopped*"
        chat.interrupted = False
    elif in_thinking or explicit_thinking:
        display = _remove_thinking_blocks(strip_prompt_markers(get_display_text(buf)))
    else:
        display = _remove_thinking_blocks(strip_prompt_markers(buf))

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
