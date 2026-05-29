import unittest


class TestModelRunner(unittest.TestCase):

    def test_interrupt_not_start_uses_sigint(self):
        """interrupt() — NOT start() — should send SIGINT (not \\x04)."""
        import inspect
        from model_lifecycle import ModelRunner

        src = inspect.getsource(ModelRunner.interrupt)
        self.assertIn("SIGINT", src)
        self.assertNotIn("\\x04", src)
        self.assertNotIn("stdin.write", src)

    def test_interrupt_uses_sigint(self):
        """interrupt() should send SIGINT (not \\x04)."""
        import inspect
        from model_lifecycle import ModelRunner

        src = inspect.getsource(ModelRunner.interrupt)
        self.assertIn("SIGINT", src)
        self.assertNotIn("\\x04", src)
        self.assertNotIn("stdin.write", src)


    def test_ctrl_d_prepends_to_first_message(self):
        """\\x04 leaked into stdin by start() prepends to the first message,
        breaking /think detection (query becomes \"\\x04/think...\").
        """
        import io
        old = __import__('sys').stdin
        try:
            __import__('sys').stdin = io.StringIO("\x04/think test\n")
            line = input()
            self.assertNotEqual(
                line, "/think test",
                "\\x04 prepended to input: /think detection fails"
            )
            self.assertEqual(
                line, "\x04/think test",
                "\\x04 leaks into first message"
            )
        finally:
            __import__('sys').stdin = old

    def test_start_no_ctrl_d_write(self):
        """ModelRunner.start() must not write \\x04 to stdin."""
        import inspect
        from model_lifecycle import ModelRunner
        src = inspect.getsource(ModelRunner.start)
        # The old code had: self.proc.stdin.write(b"\\x04")
        # After the fix, there should be no \\x04 literal in the start method
        self.assertNotIn("\\x04", src,
                         "start() must not write \\x04 — it leaks into first message")


if __name__ == "__main__":
    unittest.main()
