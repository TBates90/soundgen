"""Core mono audio buffer abstraction used across synthesis and effects."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Float32Array = NDArray[np.float32]


class AudioBuffer:
    """Immutable container for mono floating-point audio samples.

    Samples use the conventional full-scale floating-point range where ``-1.0``
    and ``1.0`` represent the negative and positive output limits. Intermediate
    DSP operations may intentionally exceed that range; call :meth:`clip` at an
    explicit boundary when hard limiting is required.
    """

    __slots__ = ("_samples", "_sample_rate")

    def __init__(self, samples: ArrayLike, sample_rate: int) -> None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("samples must be a one-dimensional mono array")
        if not np.all(np.isfinite(array)):
            raise ValueError("samples must contain only finite values")

        owned_samples = np.array(array, dtype=np.float32, copy=True, order="C")
        owned_samples.setflags(write=False)

        self._samples = owned_samples
        self._sample_rate = sample_rate

    @property
    def samples(self) -> Float32Array:
        """Read-only mono samples in ``float32`` format."""

        return self._samples

    @property
    def sample_rate(self) -> int:
        """Samples per second."""

        return self._sample_rate

    @property
    def duration(self) -> float:
        """Buffer duration in seconds."""

        return self._samples.size / self._sample_rate

    @property
    def peak(self) -> float:
        """Largest absolute sample value, or zero for an empty buffer."""

        if self._samples.size == 0:
            return 0.0
        return float(np.max(np.abs(self._samples)))

    def normalize(self) -> AudioBuffer:
        """Return a new buffer scaled so its absolute peak is ``1.0``.

        Silence remains silence rather than introducing NaN/Inf values.
        """

        peak = self.peak
        if peak == 0.0:
            return AudioBuffer(self._samples, self._sample_rate)
        return AudioBuffer(self._samples / peak, self._sample_rate)

    def apply_gain(self, gain: float) -> AudioBuffer:
        """Return a new buffer with linear gain applied to every sample."""

        try:
            gain_value = float(gain)
        except (TypeError, ValueError) as exc:
            raise ValueError("gain must be a finite number") from exc

        if not np.isfinite(gain_value):
            raise ValueError("gain must be a finite number")

        return AudioBuffer(self._samples * gain_value, self._sample_rate)

    def mix(self, other: AudioBuffer) -> AudioBuffer:
        """Mix another mono buffer sample-for-sample.

        Buffers must share a sample rate. If their lengths differ, the shorter
        buffer is treated as silence after its final sample. Mixing does not
        normalize or clip, preserving relative layer levels for later DSP.
        """

        if not isinstance(other, AudioBuffer):
            raise TypeError("other must be an AudioBuffer")
        if self._sample_rate != other._sample_rate:
            raise ValueError(
                "cannot mix buffers with different sample rates: "
                f"{self._sample_rate} != {other._sample_rate}"
            )

        sample_count = max(self._samples.size, other._samples.size)
        mixed = np.zeros(sample_count, dtype=np.float32)
        mixed[: self._samples.size] += self._samples
        mixed[: other._samples.size] += other._samples

        return AudioBuffer(mixed, self._sample_rate)

    def clip(self) -> AudioBuffer:
        """Return a new buffer hard-clipped to the full-scale ``[-1, 1]`` range."""

        return AudioBuffer(np.clip(self._samples, -1.0, 1.0), self._sample_rate)
