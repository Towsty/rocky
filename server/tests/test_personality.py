"""Rocky's character file and the speaker tuning his voice goes through."""

import unittest

import numpy as np

from brain import config, mouth, personality


class CharacterTests(unittest.TestCase):
    def test_prompt_is_complete(self):
        self.assertIn(config.HUMAN_NAME, personality.SYSTEM_PROMPT)
        self.assertIn(f"One to {config.REPLY_MAX_SENTENCES} sentences", personality.SYSTEM_PROMPT)
        self.assertIn("[neutral]", personality.SYSTEM_PROMPT)

    def test_every_canned_line_exists(self):
        for key in ("wake", "sleep", "track_on", "track_off"):
            self.assertTrue(personality.LINES.get(key), f"missing the {key} line")

    def test_voice_settings(self):
        self.assertEqual(len(config.TTS_VOICE_ID), 32)
        self.assertIn(f"hey {config.ROBOT_NAME.lower()}", config.WAKE_PHRASES)
        self.assertIn(config.ROBOT_NAME, config.STT_PROMPT)


class SpeakerTuningTests(unittest.TestCase):
    def test_equalizer_cuts_bass_and_lifts_presence(self):
        sr = mouth.SAMPLE_RATE
        t = np.arange(sr) / sr
        x = (0.3 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
        y = mouth.Equalizer(300, 6.0).process(x)
        spec_x, spec_y = np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / sr)
        bass = lambda s: s[(f > 50) & (f < 150)].sum()
        top = lambda s: s[(f > 2900) & (f < 3100)].sum()
        self.assertLess(10 * np.log10(bass(spec_y) / bass(spec_x)), -15)
        self.assertGreater(10 * np.log10(top(spec_y) / top(spec_x)), 4)

    def test_equalizer_off_is_passthrough_and_chunks_join_cleanly(self):
        x = np.random.default_rng(1).normal(0, 0.1, 4000).astype(np.float32)
        self.assertTrue(np.array_equal(mouth.Equalizer(0, 0).process(x), x))
        whole = mouth.Equalizer(300, 6.0).process(x)
        eq = mouth.Equalizer(300, 6.0)
        chunked = np.concatenate([eq.process(x[:1234]), eq.process(x[1234:])])
        self.assertTrue(np.allclose(whole, chunked, atol=1e-6))

    def test_leveler_reads_config_when_built(self):
        saved = (config.TTS_LEVEL, config.TTS_HIGHPASS_HZ, config.TTS_PRESENCE_DB)
        try:
            config.TTS_LEVEL, config.TTS_HIGHPASS_HZ, config.TTS_PRESENCE_DB = 0.22, 300.0, 6.0
            lev = mouth.Leveler()
            self.assertEqual(lev.level, 0.22)
            self.assertEqual(len(lev.eq.stages), 3)
            config.TTS_HIGHPASS_HZ, config.TTS_PRESENCE_DB = 0.0, 0.0
            self.assertEqual(len(mouth.Leveler().eq.stages), 0)
        finally:
            config.TTS_LEVEL, config.TTS_HIGHPASS_HZ, config.TTS_PRESENCE_DB = saved


if __name__ == "__main__":
    unittest.main()
