"""Turn schema + no-key behavior (Rocky first-session contract)."""

import os
import unittest
from unittest import mock

from brain.thinking import RobotBrain, _split_show
from brain.turn_schema import Turn


class ShowSplitTests(unittest.TestCase):
    def test_split_show(self):
        speak, show = _split_show("[thinking] Answer: weekend is rest day.\nSHOW: Sat-Sun cultural break.")
        self.assertIn("weekend", speak.lower())
        self.assertNotIn("SHOW", speak)
        self.assertIn("Sat-Sun", show or "")

    def test_no_show(self):
        speak, show = _split_show("Amaze! Fist my bump.")
        self.assertEqual(speak, "Amaze! Fist my bump.")
        self.assertIsNone(show)


class NoKeyTests(unittest.TestCase):
    def test_ask_without_key_is_rocky_line(self):
        env = {k: v for k, v in os.environ.items() if k != "LLM_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            brain = RobotBrain(actions={})
            turn = brain.ask("what is a weekend?")
        self.assertIsInstance(turn, Turn)
        self.assertEqual(turn.emotion, "sad")
        self.assertIn("Brain has no key", turn.speak)
        self.assertIsNone(turn.show)
        self.assertEqual(turn.acts, [])


if __name__ == "__main__":
    unittest.main()
