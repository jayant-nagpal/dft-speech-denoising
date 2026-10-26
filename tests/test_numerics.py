"""Small synthetic checks; no dataset, download or experiment is required."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from speech_dft import core


class TransformTests(unittest.TestCase):
    def test_inverse_preserves_boundaries_and_length(self):
        for length in [1, 127, 256, 431]:
            with self.subTest(length=length):
                signal = np.cos(np.arange(length) * 0.37)
                signal[-1] += 0.6
                restored = core.inverse_stft(core.stft(signal), length)
                self.assertEqual(restored.shape, signal.shape)
                np.testing.assert_allclose(restored, signal, atol=1e-12)

    def test_same_gains_preserve_source_additivity(self):
        index = np.arange(389)
        speech = np.sin(index * 0.11)
        noise = 0.3 * np.cos(index * 0.71)
        gains = np.linspace(0.05, 0.95, core.FRAME_LENGTH // 2 + 1)

        def process(signal):
            return core.inverse_stft(core.stft(signal) * gains, len(signal))

        np.testing.assert_allclose(
            process(speech + noise), process(speech) + process(noise), atol=1e-12
        )

    def test_gain_initialization_and_previous_frame(self):
        gains = core.estimate_gains(
            np.array([[1], [3], [1]], dtype=complex), np.ones(1), 0.9
        )
        first = 0.05
        xi_second = 0.9 * first**2 + 0.1 * 8
        second = xi_second / (1 + xi_second)
        xi_third = 0.9 * second**2 * 9
        expected = [first, second, xi_third / (1 + xi_third)]
        np.testing.assert_allclose(gains[:, 0], expected, atol=1e-14)

    def test_score_penalizes_speech_loss(self):
        reference = np.array([1.0, -1.0, 1.0, -1.0])
        self.assertAlmostEqual(
            core.reconstruction_snr(reference, np.zeros_like(reference)), 0
        )
        self.assertAlmostEqual(
            core.reconstruction_snr(reference, 0.5 * reference), 10 * np.log10(4)
        )

    def test_recording_is_centered_and_rms_normalized(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recording.wav"
            wavfile.write(
                path, core.SAMPLE_RATE, np.array([12, -8, 12, -8], dtype=np.int16)
            )
            signal = core.load_recording(path)
        self.assertAlmostEqual(float(signal.mean()), 0)
        self.assertAlmostEqual(float(np.sqrt(np.mean(signal**2))), 0.1)

    def test_seed_is_repeatable_and_condition_specific(self):
        arguments = ("7_jackson_10.wav", "white", 0, "test")
        self.assertEqual(core.key_seed(*arguments), core.key_seed(*arguments))
        self.assertNotEqual(
            core.key_seed(*arguments),
            core.key_seed("7_jackson_10.wav", "colored", 0, "test"),
        )


if __name__ == "__main__":
    unittest.main()
