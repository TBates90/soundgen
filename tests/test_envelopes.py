"""Tests for amplitude envelope processing."""

from __future__ import annotations

import numpy as np
import pytest

from procedural_sfx.audio import AudioBuffer
from procedural_sfx.models import EnvelopeDefinition
from procedural_sfx.synthesis import ADSREnvelope


def _ones(sample_count: int, sample_rate: int = 10) -> AudioBuffer:
    return AudioBuffer(np.ones(sample_count, dtype=np.float32), sample_rate)


def test_adsr_applies_expected_attack_decay_sustain_and_release() -> None:
    buffer = _ones(10)
    envelope = ADSREnvelope(attack=0.2, decay=0.2, sustain=0.5, release=0.2)

    result = envelope.apply(buffer)

    np.testing.assert_allclose(
        result.samples,
        np.asarray([0.0, 0.5, 1.0, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32),
        atol=1e-6,
    )


def test_attack_ramps_up_from_zero() -> None:
    result = ADSREnvelope(attack=0.4).apply(_ones(5))

    np.testing.assert_allclose(result.samples[:4], [0.0, 0.25, 0.5, 0.75], atol=1e-6)
    assert result.samples[4] == pytest.approx(1.0)


def test_sustain_level_is_held_for_remaining_samples() -> None:
    result = ADSREnvelope(attack=0.1, decay=0.1, sustain=0.25, release=0.0).apply(_ones(6))

    np.testing.assert_allclose(result.samples[2:], 0.25, atol=1e-6)


def test_release_reaches_zero() -> None:
    result = ADSREnvelope(attack=0.0, decay=0.0, sustain=0.6, release=0.3).apply(_ones(6))

    assert result.samples[-1] == pytest.approx(0.0, abs=1e-7)
    assert result.samples[-3] == pytest.approx(0.6)
    assert result.samples[-2] == pytest.approx(0.3)


def test_stages_are_proportionally_fitted_into_very_short_sound() -> None:
    buffer = _ones(3)
    envelope = ADSREnvelope(attack=0.2, decay=0.2, sustain=0.4, release=0.2)

    result = envelope.apply(buffer)

    assert result.samples.size == 3
    np.testing.assert_allclose(result.samples, [0.0, 1.0, 0.0], atol=1e-6)
    assert np.all(np.isfinite(result.samples))


def test_zero_attack_starts_at_full_amplitude_before_decay() -> None:
    result = ADSREnvelope(attack=0.0, decay=0.2, sustain=0.5, release=0.0).apply(_ones(4))

    assert result.samples[0] == pytest.approx(1.0)
    assert result.samples[-1] == pytest.approx(0.5)


def test_zero_release_does_not_force_final_sample_to_zero() -> None:
    result = ADSREnvelope(attack=0.0, decay=0.0, sustain=0.4, release=0.0).apply(_ones(4))

    np.testing.assert_allclose(result.samples, 0.4, atol=1e-6)


def test_apply_does_not_mutate_input_buffer() -> None:
    buffer = AudioBuffer(np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32), 10)
    original = buffer.samples.copy()

    result = ADSREnvelope(attack=0.2, decay=0.1, sustain=0.5, release=0.1).apply(buffer)

    np.testing.assert_array_equal(buffer.samples, original)
    assert result is not buffer


def test_from_definition_maps_configuration_values() -> None:
    definition = EnvelopeDefinition(attack=0.01, decay=0.2, sustain=0.35, release=0.4)

    envelope = ADSREnvelope.from_definition(definition)

    assert envelope.attack == pytest.approx(0.01)
    assert envelope.decay == pytest.approx(0.2)
    assert envelope.sustain == pytest.approx(0.35)
    assert envelope.release == pytest.approx(0.4)


def test_empty_buffer_remains_empty() -> None:
    buffer = AudioBuffer(np.asarray([], dtype=np.float32), 44_100)

    result = ADSREnvelope(attack=0.1, decay=0.2, sustain=0.5, release=0.3).apply(buffer)

    assert result.samples.size == 0
    assert result.sample_rate == 44_100


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attack": -0.1}, "attack"),
        ({"decay": float("inf")}, "decay"),
        ({"release": -1.0}, "release"),
        ({"sustain": -0.1}, "sustain"),
        ({"sustain": 1.1}, "sustain"),
    ],
)
def test_invalid_envelope_parameters_are_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ADSREnvelope(**kwargs)


def test_apply_rejects_non_audio_buffer() -> None:
    with pytest.raises(TypeError, match="AudioBuffer"):
        ADSREnvelope().apply(np.ones(4, dtype=np.float32))  # type: ignore[arg-type]
