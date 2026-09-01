"""Tests for the shared mono AudioBuffer abstraction."""

from __future__ import annotations

import numpy as np
import pytest

from procedural_sfx import AudioBuffer


def test_samples_are_copied_as_read_only_float32() -> None:
    source = np.array([0.25, -0.5, 1.0], dtype=np.float64)

    buffer = AudioBuffer(source, sample_rate=44_100)
    source[0] = 0.9

    assert buffer.samples.dtype == np.float32
    assert buffer.samples.tolist() == pytest.approx([0.25, -0.5, 1.0])
    assert not buffer.samples.flags.writeable

    with pytest.raises(ValueError):
        buffer.samples[0] = 0.0


def test_duration_is_derived_from_sample_count_and_rate() -> None:
    buffer = AudioBuffer(np.zeros(22_050), sample_rate=44_100)

    assert buffer.duration == pytest.approx(0.5)


def test_peak_is_largest_absolute_sample() -> None:
    buffer = AudioBuffer([0.2, -0.75, 0.5], sample_rate=44_100)

    assert buffer.peak == pytest.approx(0.75)
    assert AudioBuffer([], sample_rate=44_100).peak == 0.0


def test_apply_gain_returns_new_buffer_without_mutating_input() -> None:
    original = AudioBuffer([0.25, -0.5], sample_rate=44_100)

    amplified = original.apply_gain(2.0)

    assert original.samples.tolist() == pytest.approx([0.25, -0.5])
    assert amplified.samples.tolist() == pytest.approx([0.5, -1.0])
    assert amplified.sample_rate == original.sample_rate


def test_apply_gain_rejects_non_finite_values() -> None:
    buffer = AudioBuffer([0.25], sample_rate=44_100)

    with pytest.raises(ValueError, match="finite"):
        buffer.apply_gain(float("inf"))


def test_clip_limits_samples_to_full_scale() -> None:
    buffer = AudioBuffer([-2.0, -0.5, 0.5, 1.5], sample_rate=44_100)

    clipped = buffer.clip()

    assert clipped.samples.tolist() == pytest.approx([-1.0, -0.5, 0.5, 1.0])
    assert buffer.samples.tolist() == pytest.approx([-2.0, -0.5, 0.5, 1.5])


def test_normalize_scales_peak_to_one() -> None:
    buffer = AudioBuffer([0.25, -0.5, 0.125], sample_rate=44_100)

    normalized = buffer.normalize()

    assert normalized.peak == pytest.approx(1.0)
    assert normalized.samples.tolist() == pytest.approx([0.5, -1.0, 0.25])


def test_normalize_silence_remains_finite_silence() -> None:
    buffer = AudioBuffer(np.zeros(4), sample_rate=44_100)

    normalized = buffer.normalize()

    assert normalized.peak == 0.0
    assert np.all(np.isfinite(normalized.samples))
    assert normalized.samples.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_mix_combines_samples_without_implicit_clipping() -> None:
    left = AudioBuffer([0.75, -0.5], sample_rate=44_100)
    right = AudioBuffer([0.75, 0.25], sample_rate=44_100)

    mixed = left.mix(right)

    assert mixed.samples.tolist() == pytest.approx([1.5, -0.25])
    assert left.samples.tolist() == pytest.approx([0.75, -0.5])
    assert right.samples.tolist() == pytest.approx([0.75, 0.25])


def test_mix_zero_pads_shorter_buffer() -> None:
    long_buffer = AudioBuffer([0.5, 0.25, -0.25], sample_rate=44_100)
    short_buffer = AudioBuffer([0.25], sample_rate=44_100)

    mixed = long_buffer.mix(short_buffer)

    assert mixed.samples.tolist() == pytest.approx([0.75, 0.25, -0.25])


def test_mix_rejects_sample_rate_mismatch() -> None:
    left = AudioBuffer([0.0], sample_rate=44_100)
    right = AudioBuffer([0.0], sample_rate=48_000)

    with pytest.raises(ValueError, match="different sample rates"):
        left.mix(right)


@pytest.mark.parametrize("sample_rate", [0, -1, 44_100.0, True])
def test_invalid_sample_rates_are_rejected(sample_rate: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        AudioBuffer([0.0], sample_rate=sample_rate)  # type: ignore[arg-type]


def test_non_mono_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional mono"):
        AudioBuffer(np.zeros((2, 2)), sample_rate=44_100)


def test_non_finite_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        AudioBuffer([0.0, np.nan], sample_rate=44_100)
