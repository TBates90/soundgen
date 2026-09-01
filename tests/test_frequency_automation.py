"""Tests for oscillator frequency automation."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from procedural_sfx.models import FrequencySweepDefinition, SineGeneratorDefinition
from procedural_sfx.synthesis import (
    GeneratorFactory,
    SawGenerator,
    SineGenerator,
    SquareGenerator,
    TriangleGenerator,
)


def _expected_sine(
    *,
    start: float,
    end: float,
    curve: str,
    sample_count: int,
    sample_rate: int,
) -> np.ndarray:
    if curve == "linear":
        frequencies = np.linspace(start, end, sample_count, dtype=np.float64)
    else:
        frequencies = np.geomspace(start, end, sample_count, dtype=np.float64)

    cycles = np.zeros(sample_count, dtype=np.float64)
    if sample_count > 1:
        cycles[1:] = np.cumsum(frequencies[:-1]) / float(sample_rate)
    return np.sin(2.0 * np.pi * cycles).astype(np.float32)


def test_constant_frequency_config_remains_backward_compatible() -> None:
    definition = SineGeneratorDefinition(type="sine", frequency=440)
    generator = GeneratorFactory().create(definition)

    assert definition.frequency == pytest.approx(440)
    assert isinstance(generator, SineGenerator)

    buffer = generator.generate(duration=1.0, sample_rate=4)
    expected = SineGenerator(440).generate(duration=1.0, sample_rate=4)
    np.testing.assert_array_equal(buffer.samples, expected.samples)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (100.0, 400.0),
        (400.0, 100.0),
    ],
)
def test_linear_frequency_sweep_matches_integrated_phase(start: float, end: float) -> None:
    sample_rate = 1_000
    sample_count = 10
    sweep = FrequencySweepDefinition(start=start, end=end, curve="linear")

    buffer = SineGenerator(sweep).generate(
        duration=sample_count / sample_rate,
        sample_rate=sample_rate,
    )

    np.testing.assert_allclose(
        buffer.samples,
        _expected_sine(
            start=start,
            end=end,
            curve="linear",
            sample_count=sample_count,
            sample_rate=sample_rate,
        ),
        atol=1e-6,
    )


def test_exponential_frequency_sweep_matches_integrated_phase() -> None:
    sample_rate = 2_000
    sample_count = 12
    sweep = FrequencySweepDefinition(start=800, end=100, curve="exponential")

    buffer = SineGenerator(sweep).generate(
        duration=sample_count / sample_rate,
        sample_rate=sample_rate,
    )

    np.testing.assert_allclose(
        buffer.samples,
        _expected_sine(
            start=800,
            end=100,
            curve="exponential",
            sample_count=sample_count,
            sample_rate=sample_rate,
        ),
        atol=1e-6,
    )


def test_factory_preserves_frequency_sweep_definition() -> None:
    definition = SineGeneratorDefinition(
        type="sine",
        frequency={"start": 1_200, "end": 120, "curve": "exponential"},
    )

    assert isinstance(definition.frequency, FrequencySweepDefinition)

    generator = GeneratorFactory().create(definition)
    assert isinstance(generator, SineGenerator)
    assert generator.frequency == definition.frequency


@pytest.mark.parametrize(
    "generator_type",
    [SineGenerator, SquareGenerator, SawGenerator, TriangleGenerator],
)
def test_all_periodic_generators_share_frequency_automation(generator_type: type) -> None:
    sweep = FrequencySweepDefinition(start=100, end=300, curve="linear")
    generator = generator_type(sweep)

    buffer = generator.generate(duration=0.01, sample_rate=1_000)

    assert buffer.samples.shape == (10,)
    assert np.all(np.isfinite(buffer.samples))
    assert np.all(buffer.samples >= -1.0)
    assert np.all(buffer.samples <= 1.0)


@pytest.mark.parametrize(
    "frequency",
    [
        {"start": 0, "end": 100, "curve": "exponential"},
        {"start": 100, "end": 0, "curve": "exponential"},
        {"start": -1, "end": 100, "curve": "exponential"},
    ],
)
def test_exponential_sweep_rejects_non_positive_endpoints(frequency: dict) -> None:
    with pytest.raises(ValidationError):
        SineGeneratorDefinition(type="sine", frequency=frequency)


def test_linear_sweep_allows_zero_hz_endpoint() -> None:
    definition = SineGeneratorDefinition(
        type="sine",
        frequency={"start": 0, "end": 100, "curve": "linear"},
    )

    assert isinstance(definition.frequency, FrequencySweepDefinition)
    assert definition.frequency.start == pytest.approx(0)
