"""Procedural waveform generators backed by NumPy."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from procedural_sfx.audio import AudioBuffer
from procedural_sfx.models import (
    FrequencyDefinition,
    FrequencySweepDefinition,
    GeneratorDefinition,
    NoiseGeneratorDefinition,
    SawGeneratorDefinition,
    SineGeneratorDefinition,
    SquareGeneratorDefinition,
    TriangleGeneratorDefinition,
)


class Generator(Protocol):
    """Contract implemented by all procedural audio generators."""

    def generate(self, duration: float, sample_rate: int) -> AudioBuffer:
        """Generate a mono audio buffer for the requested duration."""
        ...


def _sample_count(duration: float, sample_rate: int) -> int:
    try:
        duration_value = float(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration must be a finite positive number") from exc

    if not np.isfinite(duration_value) or duration_value <= 0:
        raise ValueError("duration must be a finite positive number")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    count = int(round(duration_value * sample_rate))
    if count <= 0:
        raise ValueError("duration and sample_rate must produce at least one sample")
    return count


def _validate_frequency(frequency: float) -> float:
    try:
        value = float(frequency)
    except (TypeError, ValueError) as exc:
        raise ValueError("frequency must be a finite positive number") from exc

    if not np.isfinite(value) or value <= 0:
        raise ValueError("frequency must be a finite positive number")
    return value


def _frequency_values(frequency: FrequencyDefinition, sample_count: int) -> np.ndarray:
    """Resolve a constant frequency or configured sweep to per-sample values."""

    if isinstance(frequency, FrequencySweepDefinition):
        start = float(frequency.start)
        end = float(frequency.end)
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("frequency sweep endpoints must be finite")

        if frequency.curve == "linear":
            values = np.linspace(start, end, sample_count, dtype=np.float64)
        elif frequency.curve == "exponential":
            if start <= 0 or end <= 0:
                raise ValueError("exponential frequency sweeps require positive endpoints")
            values = np.geomspace(start, end, sample_count, dtype=np.float64)
        else:  # Defensive: Pydantic rejects unsupported curves before runtime.
            raise ValueError(f"unsupported frequency curve: {frequency.curve}")

        return values

    value = _validate_frequency(frequency)
    return np.full(sample_count, value, dtype=np.float64)


def _phase_cycles(frequencies: np.ndarray, sample_rate: int) -> np.ndarray:
    """Integrate instantaneous frequency into oscillator phase measured in cycles."""

    cycles = np.zeros(frequencies.size, dtype=np.float64)
    if frequencies.size > 1:
        cycles[1:] = np.cumsum(frequencies[:-1], dtype=np.float64) / float(sample_rate)
    return cycles


class _PeriodicGenerator:
    """Shared implementation for constant or frequency-automated oscillators."""

    __slots__ = ("frequency",)

    def __init__(self, frequency: FrequencyDefinition) -> None:
        if isinstance(frequency, FrequencySweepDefinition):
            self.frequency = frequency
        else:
            self.frequency = _validate_frequency(frequency)

    def generate(self, duration: float, sample_rate: int) -> AudioBuffer:
        count = _sample_count(duration, sample_rate)
        frequencies = _frequency_values(self.frequency, count)
        cycles = _phase_cycles(frequencies, sample_rate)
        samples = self._waveform(cycles)
        return AudioBuffer(np.asarray(samples, dtype=np.float32), sample_rate)

    def _waveform(self, cycles: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SineGenerator(_PeriodicGenerator):
    """Generate a sine wave with constant or automated frequency."""

    def _waveform(self, cycles: np.ndarray) -> np.ndarray:
        return np.sin(2.0 * np.pi * cycles)


class SquareGenerator(_PeriodicGenerator):
    """Generate a bipolar square wave with a 50 percent duty cycle."""

    def _waveform(self, cycles: np.ndarray) -> np.ndarray:
        phase = np.mod(cycles, 1.0)
        return np.where(phase < 0.5, 1.0, -1.0)


class SawGenerator(_PeriodicGenerator):
    """Generate a rising bipolar sawtooth wave."""

    def _waveform(self, cycles: np.ndarray) -> np.ndarray:
        phase = np.mod(cycles, 1.0)
        return (2.0 * phase) - 1.0


class TriangleGenerator(_PeriodicGenerator):
    """Generate a bipolar triangle wave."""

    def _waveform(self, cycles: np.ndarray) -> np.ndarray:
        phase = np.mod(cycles, 1.0)
        return 1.0 - (4.0 * np.abs(phase - 0.5))


class WhiteNoiseGenerator:
    """Generate uniformly distributed white noise in the full-scale range."""

    __slots__ = ("seed",)

    def __init__(self, seed: int | None = None) -> None:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer or None")
        self.seed = seed

    def generate(self, duration: float, sample_rate: int) -> AudioBuffer:
        count = _sample_count(duration, sample_rate)
        rng = np.random.default_rng(self.seed)
        samples = rng.uniform(-1.0, 1.0, size=count).astype(np.float32)
        return AudioBuffer(samples, sample_rate)


class GeneratorFactory:
    """Create concrete generators from validated configuration models."""

    def create(self, definition: GeneratorDefinition) -> Generator:
        if isinstance(definition, SineGeneratorDefinition):
            return SineGenerator(definition.frequency)
        if isinstance(definition, SquareGeneratorDefinition):
            return SquareGenerator(definition.frequency)
        if isinstance(definition, SawGeneratorDefinition):
            return SawGenerator(definition.frequency)
        if isinstance(definition, TriangleGeneratorDefinition):
            return TriangleGenerator(definition.frequency)
        if isinstance(definition, NoiseGeneratorDefinition):
            return WhiteNoiseGenerator(definition.seed)

        raise TypeError(f"unsupported generator definition: {type(definition).__name__}")
