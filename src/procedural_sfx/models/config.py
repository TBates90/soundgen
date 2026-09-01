"""Pydantic models describing procedural sound-effect configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    """Base class for strict, immutable configuration models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SineGeneratorDefinition(ConfigModel):
    type: Literal["sine"]
    frequency: float = Field(gt=0)


class SquareGeneratorDefinition(ConfigModel):
    type: Literal["square"]
    frequency: float = Field(gt=0)


class SawGeneratorDefinition(ConfigModel):
    type: Literal["saw"]
    frequency: float = Field(gt=0)


class TriangleGeneratorDefinition(ConfigModel):
    type: Literal["triangle"]
    frequency: float = Field(gt=0)


class NoiseGeneratorDefinition(ConfigModel):
    type: Literal["noise"]
    noise_type: Literal["white"] = "white"


GeneratorDefinition = Annotated[
    SineGeneratorDefinition
    | SquareGeneratorDefinition
    | SawGeneratorDefinition
    | TriangleGeneratorDefinition
    | NoiseGeneratorDefinition,
    Field(discriminator="type"),
]


class EnvelopeDefinition(ConfigModel):
    """ADSR amplitude envelope configuration.

    Attack, decay and release are expressed in seconds. Sustain is an amplitude
    ratio between zero and one.
    """

    attack: float = Field(default=0.0, ge=0)
    decay: float = Field(default=0.0, ge=0)
    sustain: float = Field(default=1.0, ge=0, le=1)
    release: float = Field(default=0.0, ge=0)


class GainEffectDefinition(ConfigModel):
    type: Literal["gain"]
    gain: float = Field(default=1.0, ge=0, le=4)


class DistortionEffectDefinition(ConfigModel):
    type: Literal["distortion"]
    drive_db: float = Field(default=0.0, ge=0, le=60)


class LowPassEffectDefinition(ConfigModel):
    type: Literal["low_pass"]
    cutoff: float = Field(gt=0)


class HighPassEffectDefinition(ConfigModel):
    type: Literal["high_pass"]
    cutoff: float = Field(gt=0)


class DelayEffectDefinition(ConfigModel):
    type: Literal["delay"]
    delay_seconds: float = Field(default=0.25, gt=0, le=5)
    feedback: float = Field(default=0.25, ge=0, lt=1)
    mix: float = Field(default=0.25, ge=0, le=1)


class ReverbEffectDefinition(ConfigModel):
    type: Literal["reverb"]
    room_size: float = Field(default=0.5, ge=0, le=1)
    damping: float = Field(default=0.5, ge=0, le=1)
    wet_level: float = Field(default=0.33, ge=0, le=1)
    dry_level: float = Field(default=0.4, ge=0, le=1)
    width: float = Field(default=1.0, ge=0, le=1)


EffectDefinition = Annotated[
    GainEffectDefinition
    | DistortionEffectDefinition
    | LowPassEffectDefinition
    | HighPassEffectDefinition
    | DelayEffectDefinition
    | ReverbEffectDefinition,
    Field(discriminator="type"),
]


class LayerDefinition(ConfigModel):
    generator: GeneratorDefinition
    envelope: EnvelopeDefinition = Field(default_factory=EnvelopeDefinition)
    volume: float = Field(default=1.0, ge=0, le=1)
    effects: list[EffectDefinition] = Field(default_factory=list)


class SoundDefinition(ConfigModel):
    """Top-level configuration object for a procedural sound effect."""

    name: str = Field(min_length=1)
    sample_rate: int = Field(default=44_100, ge=8_000, le=192_000)
    duration: float = Field(gt=0, le=60)
    master_volume: float = Field(default=1.0, ge=0, le=1)
    layers: list[LayerDefinition] = Field(min_length=1)
    effects: list[EffectDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_filter_cutoffs(self) -> SoundDefinition:
        """Reject filter cutoffs at or above the configured Nyquist frequency."""

        nyquist = self.sample_rate / 2
        effects = [*self.effects]
        for layer in self.layers:
            effects.extend(layer.effects)

        for effect in effects:
            if isinstance(effect, (LowPassEffectDefinition, HighPassEffectDefinition)):
                if effect.cutoff >= nyquist:
                    raise ValueError(
                        f"{effect.type} cutoff ({effect.cutoff:g} Hz) must be below "
                        f"the Nyquist frequency ({nyquist:g} Hz)"
                    )

        return self
