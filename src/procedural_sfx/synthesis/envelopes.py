"""Amplitude envelope processing for procedural audio buffers."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from procedural_sfx.audio import AudioBuffer
from procedural_sfx.models import EnvelopeDefinition


class Envelope(Protocol):
    """Contract implemented by amplitude envelope processors."""

    def apply(self, buffer: AudioBuffer) -> AudioBuffer:
        """Return a new buffer with the envelope applied."""
        ...


class ADSREnvelope:
    """Apply an attack/decay/sustain/release amplitude envelope.

    Attack, decay and release are expressed in seconds. Sustain is a linear
    amplitude ratio in the inclusive range ``[0, 1]``.
    """

    __slots__ = ("attack", "decay", "sustain", "release")

    def __init__(
        self,
        attack: float = 0.0,
        decay: float = 0.0,
        sustain: float = 1.0,
        release: float = 0.0,
    ) -> None:
        self.attack = _validate_non_negative("attack", attack)
        self.decay = _validate_non_negative("decay", decay)
        self.release = _validate_non_negative("release", release)
        self.sustain = _validate_sustain(sustain)

    @classmethod
    def from_definition(cls, definition: EnvelopeDefinition) -> ADSREnvelope:
        """Create an ADSR envelope from a validated configuration model."""

        if not isinstance(definition, EnvelopeDefinition):
            raise TypeError("definition must be an EnvelopeDefinition")
        return cls(
            attack=definition.attack,
            decay=definition.decay,
            sustain=definition.sustain,
            release=definition.release,
        )

    def apply(self, buffer: AudioBuffer) -> AudioBuffer:
        """Apply the envelope without mutating the input buffer."""

        if not isinstance(buffer, AudioBuffer):
            raise TypeError("buffer must be an AudioBuffer")

        sample_count = buffer.samples.size
        if sample_count == 0:
            return AudioBuffer(buffer.samples, buffer.sample_rate)

        attack_count, decay_count, release_count = _stage_sample_counts(
            sample_count=sample_count,
            sample_rate=buffer.sample_rate,
            attack=self.attack,
            decay=self.decay,
            release=self.release,
        )
        sustain_count = sample_count - attack_count - decay_count - release_count

        envelope = np.empty(sample_count, dtype=np.float32)
        cursor = 0

        if attack_count:
            envelope[cursor : cursor + attack_count] = np.linspace(
                0.0,
                1.0,
                attack_count,
                endpoint=False,
                dtype=np.float32,
            )
            cursor += attack_count

        if decay_count:
            envelope[cursor : cursor + decay_count] = np.linspace(
                1.0,
                self.sustain,
                decay_count,
                endpoint=False,
                dtype=np.float32,
            )
            cursor += decay_count

        if sustain_count:
            envelope[cursor : cursor + sustain_count] = self.sustain
            cursor += sustain_count

        if release_count:
            if release_count == 1:
                envelope[cursor] = 0.0
            else:
                envelope[cursor : cursor + release_count] = np.linspace(
                    self.sustain,
                    0.0,
                    release_count,
                    endpoint=True,
                    dtype=np.float32,
                )

        return AudioBuffer(buffer.samples * envelope, buffer.sample_rate)


def _validate_non_negative(name: str, value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc

    if not np.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return numeric


def _validate_sustain(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("sustain must be a finite number between 0 and 1") from exc

    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("sustain must be a finite number between 0 and 1")
    return numeric


def _stage_sample_counts(
    *,
    sample_count: int,
    sample_rate: int,
    attack: float,
    decay: float,
    release: float,
) -> tuple[int, int, int]:
    """Resolve ADSR stage lengths, proportionally fitting them when required."""

    requested = np.asarray(
        [attack * sample_rate, decay * sample_rate, release * sample_rate],
        dtype=np.float64,
    )
    rounded = np.rint(requested).astype(np.int64)
    rounded = np.maximum(rounded, 0)

    total = int(rounded.sum())
    if total <= sample_count:
        return tuple(int(value) for value in rounded)
    if total == 0:
        return 0, 0, 0

    scaled = rounded.astype(np.float64) * (sample_count / total)
    fitted = np.floor(scaled).astype(np.int64)
    remainder = sample_count - int(fitted.sum())

    if remainder:
        fractional = scaled - fitted
        eligible = np.flatnonzero(rounded > 0)
        order = eligible[np.argsort(-fractional[eligible], kind="stable")]
        for index in order[:remainder]:
            fitted[index] += 1

    return tuple(int(value) for value in fitted)
