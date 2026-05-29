import unittest


class TestModelRunner(unittest.TestCase):

    def test_no_ctrl_d_in_start(self):
        """start() should NOT write \\x04 to stdin (uses SIGINT via interrupt() instead)."""
        import inspect
        from model_lifecycle import ModelRunner

        src = inspect.getsource(ModelRunner.start)
        self.assertNotIn(
            b"\\x04",
            src.encode(),
            "start() must not write \\x04 — chat.py no longer handles Ctrl-D "
            "via stdin. The byte would bleed into input() and corrupt the "
            "first user response.",
        )

    def test_interrupt_uses_sigint(self):
        """interrupt() should send SIGINT (not \\x04)."""
        import inspect
        from model_lifecycle import ModelRunner

        src = inspect.getsource(ModelRunner.interrupt)
        self.assertIn("SIGINT", src)
        self.assertNotIn("\\x04", src)
        self.assertNotIn("stdin.write", src)


if __name__ == "__main__":
    unittest.main()
