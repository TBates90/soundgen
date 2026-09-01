"""Tests for runtime audio effects and the effect factory."""

from __future__ import annotations

import numpy as np
import pytest

from procedural_sfx.audio import AudioBuffer
from procedural_sfx.effects import (
    DelayEffect,
    DistortionEffect,
    EffectFactory,
    GainEffect,
    HighPassEffect,
    LowPassEffect,
    ReverbEffect,
)
from procedural_sfx.models import (
    DelayEffectDefinition,
    DistortionEffectDefinition,
    GainEffectDefinition,
    HighPassEffectDefinition,
    LowPassEffectDefinition,
    ReverbEffectDefinition,
)


def _impulse(sample_count: int = 256, sample_rate: int = 8_000) -> AudioBuffer:
    samples = np.zeros(sample_count, dtype=np.float32)
    samples[0] = 0.5
    return AudioBuffer(samples, sample_rate)


@pytest.mark.parametrize(
    ("definition", "expected_type"),
    [
        (GainEffectDefinition(type="gain", gain=0.5), GainEffect),
        (DistortionEffectDefinition(type="distortion", drive_db=12), DistortionEffect),
        (LowPassEffectDefinition(type="low_pass", cutoff=1_000), LowPassEffect),
        (HighPassEffectDefinition(type="high_pass", cutoff=120), HighPassEffect),
        (
            DelayEffectDefinition(
                type="delay",
                delay_seconds=0.01,
                feedback=0.3,
                mix=0.4,
            ),
            DelayEffect,
        ),
        (
            ReverbEffectDefinition(
                type="reverb",
                room_size=0.3,
                damping=0.4,
                wet_level=0.2,
                dry_level=0.8,
                width=1.0,
            ),
            ReverbEffect,
        ),
    ],
)
def test_effect_factory_dispatches_definition(
    definition: object,
    expected_type: type,
) -> None:
    effect = EffectFactory().create(definition)  # type: ignore[arg-type]

    assert isinstance(effect, expected_type)


def test_gain_effect_applies_linear_gain_without_mutating_input() -> None:
    source = AudioBuffer(np.array([-0.8, -0.25, 0.0, 0.5], dtype=np.float32), 44_100)
    before = source.samples.copy()

    output = GainEffect(0.5).process(source)

    np.testing.assert_array_equal(source.samples, before)
    np.testing.assert_allclose(
        output.samples,
        np.array([-0.4, -0.125, 0.0, 0.25], dtype=np.float32),
    )
    assert output.sample_rate == source.sample_rate


@pytest.mark.parametrize(
    "effect",
    [
        DistortionEffect(12),
        LowPassEffect(1_000),
        HighPassEffect(120),
        DelayEffect(delay_seconds=0.01, feedback=0.3, mix=0.4),
        ReverbEffect(
            room_size=0.3,
            damping=0.4,
            wet_level=0.2,
            dry_level=0.8,
            width=1.0,
        ),
    ],
)
def test_pedalboard_effects_preserve_audio_buffer_contract(effect: object) -> None:
    source = _impulse()
    before = source.samples.copy()

    output = effect.process(source)  # type: ignore[attr-defined]

    np.testing.assert_array_equal(source.samples, before)
    assert output.samples.shape == source.samples.shape
    assert output.samples.dtype == np.float32
    assert output.sample_rate == source.sample_rate
    assert np.all(np.isfinite(output.samples))


def test_distortion_changes_nonzero_signal() -> None:
    source = AudioBuffer(np.full(128, 0.5, dtype=np.float32), 8_000)

    output = DistortionEffect(18).process(source)

    assert not np.array_equal(output.samples, source.samples)


@pytest.mark.parametrize(
    "effect",
    [
        DelayEffect(delay_seconds=0.005, feedback=0.4, mix=0.5),
        ReverbEffect(
            room_size=0.6,
            damping=0.3,
            wet_level=0.5,
            dry_level=0.5,
            width=1.0,
        ),
    ],
)
def test_stateful_effects_reset_between_independent_process_calls(effect: object) -> None:
    source = _impulse(sample_count=512, sample_rate=8_000)

    first = effect.process(source)  # type: ignore[attr-defined]
    second = effect.process(source)  # type: ignore[attr-defined]

    np.testing.assert_array_equal(first.samples, second.samples)


def test_factory_rejects_unknown_definition() -> None:
    with pytest.raises(TypeError, match="unsupported effect definition"):
        EffectFactory().create(object())  # type: ignore[arg-type]


def test_effect_rejects_non_audio_buffer() -> None:
    with pytest.raises(TypeError, match="AudioBuffer"):
        GainEffect(1.0).process(np.zeros(8, dtype=np.float32))  # type: ignore[arg-type]
