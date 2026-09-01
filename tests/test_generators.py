"""Tests for NumPy-backed procedural waveform generators."""

from __future__ import annotations

import numpy as np
import pytest

from procedural_sfx.models import (
    NoiseGeneratorDefinition,
    SawGeneratorDefinition,
    SineGeneratorDefinition,
    SquareGeneratorDefinition,
    TriangleGeneratorDefinition,
)
from procedural_sfx.synthesis import (
    GeneratorFactory,
    SawGenerator,
    SineGenerator,
    SquareGenerator,
    TriangleGenerator,
    WhiteNoiseGenerator,
)


@pytest.mark.parametrize(
    "generator",
    [
        SineGenerator(220),
        SquareGenerator(220),
        SawGenerator(220),
        TriangleGenerator(220),
        WhiteNoiseGenerator(seed=1234),
    ],
)
def test_generators_produce_normalized_float32_audio(generator: object) -> None:
    buffer = generator.generate(duration=0.01, sample_rate=1_000)  # type: ignore[attr-defined]

    assert buffer.samples.shape == (10,)
    assert buffer.samples.dtype == np.float32
    assert np.all(buffer.samples >= -1.0)
    assert np.all(buffer.samples <= 1.0)


def test_sine_generator_matches_expected_samples() -> None:
    buffer = SineGenerator(1).generate(duration=1.0, sample_rate=4)

    np.testing.assert_allclose(
        buffer.samples,
        np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("definition", "expected_type"),
    [
        (SineGeneratorDefinition(type="sine", frequency=440), SineGenerator),
        (SquareGeneratorDefinition(type="square", frequency=440), SquareGenerator),
        (SawGeneratorDefinition(type="saw", frequency=440), SawGenerator),
        (TriangleGeneratorDefinition(type="triangle", frequency=440), TriangleGenerator),
        (
            NoiseGeneratorDefinition(type="noise", noise_type="white", seed=99),
            WhiteNoiseGenerator,
        ),
    ],
)
def test_generator_factory_dispatches_config(
    definition: object,
    expected_type: type,
) -> None:
    generator = GeneratorFactory().create(definition)  # type: ignore[arg-type]

    assert isinstance(generator, expected_type)


def test_seeded_white_noise_is_deterministic() -> None:
    generator = WhiteNoiseGenerator(seed=12345)

    first = generator.generate(duration=0.05, sample_rate=1_000)
    second = generator.generate(duration=0.05, sample_rate=1_000)

    np.testing.assert_array_equal(first.samples, second.samples)


def test_different_noise_seeds_produce_different_samples() -> None:
    first = WhiteNoiseGenerator(seed=1).generate(duration=0.05, sample_rate=1_000)
    second = WhiteNoiseGenerator(seed=2).generate(duration=0.05, sample_rate=1_000)

    assert not np.array_equal(first.samples, second.samples)


def test_factory_preserves_noise_seed() -> None:
    definition = NoiseGeneratorDefinition(type="noise", seed=314159)
    generator = GeneratorFactory().create(definition)

    assert isinstance(generator, WhiteNoiseGenerator)
    assert generator.seed == 314159


def test_generator_rejects_invalid_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        SineGenerator(440).generate(duration=0, sample_rate=44_100)


def test_generator_rejects_invalid_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        SineGenerator(440).generate(duration=0.1, sample_rate=0)


def test_white_noise_rejects_invalid_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        WhiteNoiseGenerator(seed=-1)
