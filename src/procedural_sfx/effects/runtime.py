"""Runtime audio effects and Pedalboard adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pedalboard import (
    Delay,
    Distortion,
    HighpassFilter,
    LowpassFilter,
    Pedalboard,
    Reverb,
)

from procedural_sfx.audio import AudioBuffer
from procedural_sfx.models import (
    DelayEffectDefinition,
    DistortionEffectDefinition,
    EffectDefinition,
    GainEffectDefinition,
    HighPassEffectDefinition,
    LowPassEffectDefinition,
    ReverbEffectDefinition,
)


class AudioEffect(Protocol):
    """Contract implemented by runtime audio effects."""

    def process(self, buffer: AudioBuffer) -> AudioBuffer:
        """Return a processed copy of the supplied audio buffer."""
        ...


class GainEffect:
    """Apply linear gain using the native AudioBuffer operation."""

    __slots__ = ("gain",)

    def __init__(self, gain: float) -> None:
        self.gain = float(gain)

    def process(self, buffer: AudioBuffer) -> AudioBuffer:
        _validate_buffer(buffer)
        return buffer.apply_gain(self.gain)


class _PedalboardEffect:
    """Base adapter that keeps Pedalboard details inside the effects package."""

    __slots__ = ("_plugin",)

    def __init__(self, plugin: object) -> None:
        self._plugin = plugin

    def process(self, buffer: AudioBuffer) -> AudioBuffer:
        _validate_buffer(buffer)

        # ``reset=True`` prevents stateful effects such as delay/reverb from
        # leaking history between independent renders that reuse one effect.
        processed = Pedalboard([self._plugin])(
            buffer.samples,
            float(buffer.sample_rate),
            reset=True,
        )
        samples = np.asarray(processed, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("Pedalboard effect returned non-mono audio")
        return AudioBuffer(samples, buffer.sample_rate)


class DistortionEffect(_PedalboardEffect):
    """Apply Pedalboard's tanh-based distortion."""

    def __init__(self, drive_db: float) -> None:
        super().__init__(Distortion(drive_db=float(drive_db)))


class LowPassEffect(_PedalboardEffect):
    """Apply a first-order low-pass filter."""

    def __init__(self, cutoff: float) -> None:
        super().__init__(LowpassFilter(cutoff_frequency_hz=float(cutoff)))


class HighPassEffect(_PedalboardEffect):
    """Apply a first-order high-pass filter."""

    def __init__(self, cutoff: float) -> None:
        super().__init__(HighpassFilter(cutoff_frequency_hz=float(cutoff)))


class DelayEffect(_PedalboardEffect):
    """Apply a digital delay while preserving the input buffer length."""

    def __init__(self, delay_seconds: float, feedback: float, mix: float) -> None:
        super().__init__(
            Delay(
                delay_seconds=float(delay_seconds),
                feedback=float(feedback),
                mix=float(mix),
            )
        )


class ReverbEffect(_PedalboardEffect):
    """Apply Pedalboard's FreeVerb-derived reverb."""

    def __init__(
        self,
        room_size: float,
        damping: float,
        wet_level: float,
        dry_level: float,
        width: float,
    ) -> None:
        super().__init__(
            Reverb(
                room_size=float(room_size),
                damping=float(damping),
                wet_level=float(wet_level),
                dry_level=float(dry_level),
                width=float(width),
            )
        )


class EffectFactory:
    """Create runtime effects from validated effect definitions."""

    def create(self, definition: EffectDefinition) -> AudioEffect:
        if isinstance(definition, GainEffectDefinition):
            return GainEffect(definition.gain)
        if isinstance(definition, DistortionEffectDefinition):
            return DistortionEffect(definition.drive_db)
        if isinstance(definition, LowPassEffectDefinition):
            return LowPassEffect(definition.cutoff)
        if isinstance(definition, HighPassEffectDefinition):
            return HighPassEffect(definition.cutoff)
        if isinstance(definition, DelayEffectDefinition):
            return DelayEffect(
                delay_seconds=definition.delay_seconds,
                feedback=definition.feedback,
                mix=definition.mix,
            )
        if isinstance(definition, ReverbEffectDefinition):
            return ReverbEffect(
                room_size=definition.room_size,
                damping=definition.damping,
                wet_level=definition.wet_level,
                dry_level=definition.dry_level,
                width=definition.width,
            )

        raise TypeError(f"unsupported effect definition: {type(definition).__name__}")


def _validate_buffer(buffer: AudioBuffer) -> None:
    if not isinstance(buffer, AudioBuffer):
        raise TypeError("buffer must be an AudioBuffer")
