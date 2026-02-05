"""Unit tests for audio utilities in pipeline.py.

Requires numpy but not torch/soundfile — mocks are used for I/O where needed.
"""

import numpy as np
import pytest

from tts_pipeline.pipeline import _peak_normalize


class TestPeakNormalize:
    """Tests for the peak-normalisation helper."""

    def test_normalises_to_target(self):
        audio = np.array([0.0, 0.5, -1.0, 0.25], dtype=np.float32)
        result = _peak_normalize(audio, target=0.9)
        assert np.isclose(np.abs(result).max(), 0.9, atol=1e-6)

    def test_preserves_sign(self):
        audio = np.array([0.5, -1.0, 0.3], dtype=np.float32)
        result = _peak_normalize(audio, target=0.8)
        # The most-negative sample should still be negative
        assert result[1] < 0

    def test_relative_amplitudes_preserved(self):
        audio = np.array([0.2, 0.4, 0.8], dtype=np.float32)
        result = _peak_normalize(audio, target=0.9)
        # Ratios should be unchanged
        assert np.isclose(result[0] / result[1], audio[0] / audio[1], atol=1e-5)
        assert np.isclose(result[1] / result[2], audio[1] / audio[2], atol=1e-5)

    def test_silent_audio_unchanged(self):
        audio = np.zeros(100, dtype=np.float32)
        result = _peak_normalize(audio, target=0.9)
        np.testing.assert_array_equal(result, audio)

    def test_single_sample(self):
        audio = np.array([0.5], dtype=np.float32)
        result = _peak_normalize(audio, target=1.0)
        assert np.isclose(result[0], 1.0, atol=1e-6)

    def test_already_at_target(self):
        audio = np.array([0.0, 0.9, -0.45], dtype=np.float32)
        result = _peak_normalize(audio, target=0.9)
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_output_shape_matches_input(self):
        audio = np.random.uniform(-0.5, 0.5, size=4096).astype(np.float32)
        result = _peak_normalize(audio, target=0.85)
        assert result.shape == audio.shape

    def test_default_target_is_point_nine(self):
        audio = np.array([2.0, -1.0], dtype=np.float32)
        result = _peak_normalize(audio)  # default target
        assert np.isclose(np.abs(result).max(), 0.9, atol=1e-6)
