"""setup's menu colours: only on a terminal that takes them.

    python tools/test_setup_colours.py
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402


class Colours(unittest.TestCase):
    def asked(self, tty=True, win=False, env=None):
        with mock.patch.object(sys.stdout, "isatty", lambda: tty, create=True), \
                mock.patch.object(setup, "WIN", win), mock.patch.dict(os.environ, env or {}, clear=True):
            return setup.colours()

    def test_a_terminal_gets_them(self):
        self.assertTrue(self.asked())

    def test_redirected_output_gets_none(self):
        self.assertFalse(self.asked(tty=False))

    def test_no_color_and_a_dumb_terminal_get_none(self):
        self.assertFalse(self.asked(env={"NO_COLOR": "1"}))
        self.assertFalse(self.asked(env={"TERM": "dumb"}))

    def test_a_windows_console_that_refuses_gets_none(self):
        k32 = mock.Mock()
        k32.GetConsoleMode.return_value = 1
        k32.SetConsoleMode.return_value = 0            # an old console: no virtual terminal processing
        with mock.patch("ctypes.windll", mock.Mock(kernel32=k32), create=True):
            self.assertFalse(self.asked(win=True))
        k32.SetConsoleMode.return_value = 1
        with mock.patch("ctypes.windll", mock.Mock(kernel32=k32), create=True):
            self.assertTrue(self.asked(win=True))


if __name__ == "__main__":
    unittest.main()
