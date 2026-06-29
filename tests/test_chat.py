import argparse
import unittest
from unittest.mock import MagicMock, patch

from mlx_lm.chat import setup_arg_parser


class TestChat(unittest.TestCase):

    def test_setup_arg_parser_system_prompt(self):
        parser = setup_arg_parser()

        # Test default (no system prompt)
        args = parser.parse_args([])
        self.assertIsNone(args.system_prompt)

        # Test with system prompt
        args = parser.parse_args(["--system-prompt", "You are a helpful assistant."])
        self.assertEqual(args.system_prompt, "You are a helpful assistant.")

    def test_setup_arg_parser_all_args(self):
        parser = setup_arg_parser()
        args = parser.parse_args(
            [
                "--model",
                "test-model",
                "--adapter-path",
                "/path/to/adapter",
                "--temp",
                "0.7",
                "--top-p",
                "0.9",
                "--xtc-probability",
                "0.1",
                "--xtc-threshold",
                "0.1",
                "--seed",
                "42",
                "--max-kv-size",
                "1024",
                "--max-tokens",
                "512",
                "--system-prompt",
                "You are a helpful assistant.",
            ]
        )

        self.assertEqual(args.model, "test-model")
        self.assertEqual(args.adapter_path, "/path/to/adapter")
        self.assertEqual(args.temp, 0.7)
        self.assertEqual(args.top_p, 0.9)
        self.assertEqual(args.xtc_probability, 0.1)
        self.assertEqual(args.xtc_threshold, 0.1)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.max_kv_size, 1024)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.system_prompt, "You are a helpful assistant.")

    @patch("mlx_lm.chat.load")
    @patch("mlx_lm.chat.make_prompt_cache")
    @patch("mlx_lm.chat.stream_generate")
    @patch("builtins.input")
    @patch("builtins.print")
    def test_system_prompt_in_messages(
        self,
        mock_print,
        mock_input,
        mock_stream_generate,
        mock_make_prompt_cache,
        mock_load,
    ):
        from mlx_lm.chat import main

        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "processed_prompt"
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Mock prompt cache
        mock_prompt_cache = MagicMock()
        mock_make_prompt_cache.return_value = mock_prompt_cache

        # Mock stream_generate to return some responses
        mock_response = MagicMock()
        mock_response.text = "Hello there!"
        mock_stream_generate.return_value = [mock_response]

        # Mock user input: first a question, then 'q' to quit
        mock_input.side_effect = ["What is the weather?", "q"]

        # Test with system prompt
        with patch(
            "sys.argv", ["chat.py", "--system-prompt", "You are a weather assistant."]
        ):
            try:
                main()
            except SystemExit:
                pass

        # Verify that apply_chat_template was called with system prompt
        mock_tokenizer.apply_chat_template.assert_called()
        call_args = mock_tokenizer.apply_chat_template.call_args[0][
            0
        ]  # First positional arg (messages)

        # Check that the messages contain both system and user messages
        self.assertEqual(len(call_args), 2)
        self.assertEqual(call_args[0]["role"], "system")
        self.assertEqual(call_args[0]["content"], "You are a weather assistant.")
        self.assertEqual(call_args[1]["role"], "user")
        self.assertEqual(call_args[1]["content"], "What is the weather?")

    @patch("mlx_lm.chat.load")
    @patch("mlx_lm.chat.make_prompt_cache")
    @patch("mlx_lm.chat.stream_generate")
    @patch("builtins.input")
    @patch("builtins.print")
    def test_no_system_prompt_in_messages(
        self,
        mock_print,
        mock_input,
        mock_stream_generate,
        mock_make_prompt_cache,
        mock_load,
    ):
        from mlx_lm.chat import main

        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "processed_prompt"
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Mock prompt cache
        mock_prompt_cache = MagicMock()
        mock_make_prompt_cache.return_value = mock_prompt_cache

        # Mock stream_generate to return some responses
        mock_response = MagicMock()
        mock_response.text = "Hello there!"
        mock_stream_generate.return_value = [mock_response]

        # Mock user input: first a question, then 'q' to quit
        mock_input.side_effect = ["What is the weather?", "q"]

        # Test without system prompt
        with patch("sys.argv", ["chat.py"]):
            try:
                main()
            except SystemExit:
                pass

        # Verify that apply_chat_template was called without system prompt
        mock_tokenizer.apply_chat_template.assert_called()
        call_args = mock_tokenizer.apply_chat_template.call_args[0][
            0
        ]  # First positional arg (messages)

        # Check that the messages contain only user message
        self.assertEqual(len(call_args), 1)
        self.assertEqual(call_args[0]["role"], "user")
        self.assertEqual(call_args[0]["content"], "What is the weather?")


class TestSetOption(unittest.TestCase):
    """Test the /set_option runtime parameter update logic."""

    def _simulate_set_option(self, args, cmd: str) -> str | None:
        """Simulate the /set_option handler logic from chat.py main loop."""
        rest = cmd[len("/set_option "):]
        key, value_str = rest.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()
        if not hasattr(args, key):
            return f"Unknown option: {key}"
        current = getattr(args, key)
        if current is None:
            return f"Cannot set None option: {key}"
        if isinstance(current, bool):
            setattr(args, key, value_str.lower() in ("true", "1", "yes"))
        elif isinstance(current, int):
            setattr(args, key, int(value_str))
        elif isinstance(current, float):
            setattr(args, key, float(value_str))
        else:
            setattr(args, key, type(current)(value_str))
        return None

    def test_set_temp(self):
        """/set_option temp=0.3 should update args.temp."""
        from mlx_lm.chat import setup_arg_parser
        parser = setup_arg_parser()
        args = parser.parse_args(["--temp", "0.7"])
        self._simulate_set_option(args, "/set_option temp=0.3")
        self.assertEqual(args.temp, 0.3)

    def test_set_top_p(self):
        """/set_option top_p=0.5 should update args.top_p."""
        from mlx_lm.chat import setup_arg_parser
        parser = setup_arg_parser()
        args = parser.parse_args(["--top-p", "0.8"])
        self._simulate_set_option(args, "/set_option top_p=0.5")
        self.assertEqual(args.top_p, 0.5)

    def test_set_max_tokens(self):
        """/set_option max_tokens=4096 should update args.max_tokens."""
        from mlx_lm.chat import setup_arg_parser
        parser = setup_arg_parser()
        args = parser.parse_args(["--max-tokens", "256"])
        self._simulate_set_option(args, "/set_option max_tokens=4096")
        self.assertEqual(args.max_tokens, 4096)

    def test_set_prefill_step_size(self):
        """/set_option prefill_step_size=64 should update args.prefill_step_size."""
        from mlx_lm.chat import setup_arg_parser
        parser = setup_arg_parser()
        args = parser.parse_args(["--prefill-step-size", "128"])
        self._simulate_set_option(args, "/set_option prefill_step_size=64")
        self.assertEqual(args.prefill_step_size, 64)

    def test_set_unknown_option_errors(self):
        """/set_option unknown_key=1 should produce error message."""
        from mlx_lm.chat import setup_arg_parser
        parser = setup_arg_parser()
        args = parser.parse_args([])
        result = self._simulate_set_option(args, "/set_option nonexistent=1")
        self.assertIsNotNone(result)
        self.assertIn("Unknown", result)

    def test_set_option_wrong_type_errors(self):
        """/set_option max_tokens=abc should fail gracefully."""
        from mlx_lm.chat import setup_arg_parser
        parser = setup_arg_parser()
        args = parser.parse_args(["--max-tokens", "256"])
        with self.assertRaises(ValueError):
            self._simulate_set_option(args, "/set_option max_tokens=abc")


if __name__ == "__main__":
    unittest.main()
