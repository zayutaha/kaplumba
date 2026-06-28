import asyncio
import unittest
from unittest.mock import MagicMock

from conversation_engine import (
    _clean_raw_text,
    _detect_thinking_start,
    _has_thinking_end,
    _remove_thinking_blocks,
)


def make_mock_chat():
    """Build a ChatUI mock that conversation_engine can call into."""
    chat = MagicMock()
    chat.interrupted = False
    chat.first_message = False
    chat._on_crash = None

    # query_one("#chat") → VerticalScroll with scroll_offset etc.
    class FakeScrollOffset:
        y = 0
    class FakeVirtualSize:
        height = 20
    class FakeRegion:
        height = 20
    class FakeWidget:
        scroll_offset = FakeScrollOffset()
        virtual_size = FakeVirtualSize()
        region = FakeRegion()
        def scroll_end(self, animate=False):
            pass
    chat.query_one.return_value = FakeWidget()

    chat.handle_stream_chunk = AsyncMock()
    chat.handle_stream_finished = AsyncMock()
    chat._set_busy = MagicMock()
    return chat


class AsyncMock(MagicMock):
    """MagicMock that works with async functions."""
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def make_mock_port(chunks: list[str]):
    port = MagicMock()
    port.running = True

    async def send_message(text):
        for c in chunks:
            yield c
    port.send_message = send_message
    return port


class TestThinkingTagDetection(unittest.TestCase):

    def test_detect_qwen_think(self):
        self.assertTrue(_detect_thinking_start("<think>"))
        self.assertTrue(_has_thinking_end("</think>"))

    def test_detect_gemma4_channel(self):
        self.assertTrue(_detect_thinking_start("<|channel>"))
        self.assertTrue(_detect_thinking_start("<|channel>thought\n"))
        self.assertFalse(_detect_thinking_start("regular text"))

    def test_has_channel_end(self):
        self.assertTrue(_has_thinking_end("<channel|>"))
        self.assertTrue(_has_thinking_end("stuff<channel|>more"))
        self.assertFalse(_has_thinking_end("no close here"))

    def test_remove_qwen_think_block(self):
        result = _remove_thinking_blocks("prefix<think>hidden</think>visible")
        self.assertEqual(result, "prefixvisible")

    def test_remove_gemma4_channel_block(self):
        result = _remove_thinking_blocks(
            "before<|channel>thought\nhidden<channel|>after"
        )
        self.assertEqual(result, "beforeafter")

    def test_remove_multiple_think_blocks(self):
        result = _remove_thinking_blocks(
            "a<think>1</think>b<think>2</think>c"
        )
        self.assertEqual(result, "abc")

    def test_remove_no_tags_unchanged(self):
        t = "hello world"
        self.assertEqual(_remove_thinking_blocks(t), t)


class TestGemma4Streaming(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.chat = make_mock_chat()

    async def test_gemma4_response_appears_in_final(self):
        """Response after <channel|> should appear in final output."""
        from conversation_engine import run_model_stream

        port = make_mock_port([
            "<|channel>thought\n",
            "hidden thinking\n",
            "<channel|>",
            "visible answer",
        ])
        await run_model_stream(self.chat, port, "/think test")
        final = self.chat.handle_stream_finished.call_args[0][0]
        self.assertIn("visible answer", final)
        self.assertNotIn("hidden thinking", final)

    async def test_gemma4_close_and_response_same_chunk(self):
        """Closing tag and response in same chunk: response must appear."""
        from conversation_engine import run_model_stream

        port = make_mock_port([
            "<|channel>thought\nhidden<channel|>visible response",
        ])
        await run_model_stream(self.chat, port, "question")
        final = self.chat.handle_stream_finished.call_args[0][0]
        self.assertIn(
            "visible response", final,
            "Response must appear even when in same chunk as closing tag",
        )
        self.assertNotIn("hidden", final)

    async def test_gemma4_no_thinking_tags(self):
        """Without thinking tags, all text renders normally."""
        from conversation_engine import run_model_stream

        port = make_mock_port([
            "Hello, this is a normal response.",
        ])
        await run_model_stream(self.chat, port, "question")
        final = self.chat.handle_stream_finished.call_args[0][0]
        self.assertIn("Hello", final)

    async def test_gemma4_thinking_not_started_with_turn_token(self):
        """<|turn>model prefix before thinking should not trigger detection."""
        from conversation_engine import run_model_stream

        port = make_mock_port([
            "<|turn>model\n",
            "<|channel>thought\n",
            "hidden<channel|>",
            "the final answer",
        ])
        await run_model_stream(self.chat, port, "question")
        final = self.chat.handle_stream_finished.call_args[0][0]
        self.assertIn("final answer", final)
        self.assertNotIn("hidden", final)

    async def test_gemma4_multiple_channel_blocks(self):
        """Multiple <|channel> blocks: only last response kept."""
        from conversation_engine import run_model_stream

        port = make_mock_port([
            "<|channel>thought\n",
            "first think<channel|>",
            "first response",
            "<|channel>thought\n",
            "second think<channel|>",
            "second response",
        ])
        await run_model_stream(self.chat, port, "question")
        final = self.chat.handle_stream_finished.call_args[0][0]
        # Both responses should appear (regex handles multi-block)
        self.assertIn("first response", final)
        self.assertIn("second response", final)

    async def test_gemma4_explicit_thinking_all_inside_block(self):
        """When /think is used and entire response is inside <|channel>...<channel|>."""
        from conversation_engine import run_model_stream

        port = make_mock_port([
            "<|channel>thought\n",
            "The answer is 42.\n",
            "<channel|>",
        ])
        await run_model_stream(self.chat, port, "/think test")
        final = self.chat.handle_stream_finished.call_args[0][0]
        self.assertIn(
            "42", final,
            "Answer inside thinking block must appear in final output",
        )

    async def test_gemma4_auto_thinking_all_inside_block(self):
        """Auto-detected thinking where everything is inside tags."""
        from conversation_engine import run_model_stream

        port = make_mock_port([
            "<|channel>thought\n",
            "hidden thinking\n",
            "the answer is 42.\n",
            "<channel|>",
        ])
        await run_model_stream(self.chat, port, "question")
        final = self.chat.handle_stream_finished.call_args[0][0]
        self.assertIn(
            "42", final,
            "Answer inside thinking block must appear even without /think",
        )

    async def test_gemma4_explicit_thinking_stream_inside_block(self):
        """During streaming with /think, content inside block shows after close."""
        from conversation_engine import run_model_stream

        port = make_mock_port([
            "<|channel>thought\n",
            "thinking<channel|>",
        ])
        await run_model_stream(self.chat, port, "/think test")
        final = self.chat.handle_stream_finished.call_args[0][0]
        # The thinking content should be shown (it's inside the block but
        # nothing follows the close tag)
        self.assertTrue(
            len(final) > 0,
            "Should show content when nothing follows close tag",
        )

    async def test_gemma4_chunk_streams_during_thinking_shows_no_content(self):
        """During thinking, no response text should be streamed to chat."""
        from conversation_engine import run_model_stream

        port = make_mock_port([
            "<|channel>thought\n",
            "still thinking\n",
            "<channel|>",
        ])
        await run_model_stream(self.chat, port, "question")
        # handle_stream_chunk should NOT have been called with thinking content
        for call_args in self.chat.handle_stream_chunk.call_args_list:
            args = call_args[0]
            self.assertNotIn(
                "still thinking", args[0],
                "Thinking content should not be streamed to chat",
            )


class TestCleanRawText(unittest.TestCase):

    def test_strips_info_lines(self):
        """_clean_raw_text should remove lines starting with [INFO]."""
        raw = "Hello world\n[INFO] Generation stopped.\nMore text"
        cleaned = _clean_raw_text(raw)
        self.assertNotIn("[INFO]", cleaned)

    def test_preserves_normal_lines(self):
        """_clean_raw_text should keep non-[INFO] lines."""
        raw = "Hello world\nMore text"
        cleaned = _clean_raw_text(raw)
        self.assertEqual(cleaned, "Hello world\nMore text")

    def test_empty_string(self):
        """_clean_raw_text should handle empty string."""
        self.assertEqual(_clean_raw_text(""), "")


if __name__ == "__main__":
    unittest.main()
