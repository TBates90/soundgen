"""Tests for JSON sound-effect configuration models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from procedural_sfx.models import (
    DelayEffectDefinition,
    DistortionEffectDefinition,
    GainEffectDefinition,
    HighPassEffectDefinition,
    LowPassEffectDefinition,
    ReverbEffectDefinition,
    SineGeneratorDefinition,
    SoundDefinition,
)


def valid_config() -> dict:
    return {
        "name": "laser_01",
        "sample_rate": 44_100,
        "duration": 0.4,
        "master_volume": 0.8,
        "layers": [
            {
                "generator": {"type": "sine", "frequency": 900},
                "envelope": {
                    "attack": 0.001,
                    "decay": 0.08,
                    "sustain": 0.3,
                    "release": 0.1,
                },
                "volume": 1.0,
                "effects": [],
            }
        ],
        "effects": [],
    }


def test_valid_config_parses_to_strongly_typed_definition() -> None:
    sound = SoundDefinition.model_validate(valid_config())

    assert sound.name == "laser_01"
    assert sound.sample_rate == 44_100
    assert sound.duration == pytest.approx(0.4)
    assert isinstance(sound.layers[0].generator, SineGeneratorDefinition)
    assert sound.layers[0].generator.frequency == pytest.approx(900)


def test_invalid_generator_type_is_rejected() -> None:
    config = valid_config()
    config["layers"][0]["generator"] = {"type": "supersaw", "frequency": 440}

    with pytest.raises(ValidationError, match="generator"):
        SoundDefinition.model_validate(config)


def test_negative_duration_is_rejected() -> None:
    config = valid_config()
    config["duration"] = -0.5

    with pytest.raises(ValidationError, match="duration"):
        SoundDefinition.model_validate(config)


def test_json_parsing() -> None:
    sound = SoundDefinition.model_validate_json(json.dumps(valid_config()))

    assert sound.name == "laser_01"
    assert len(sound.layers) == 1


def test_multiple_layers_parse_independently() -> None:
    config = valid_config()
    config["layers"].append(
        {
            "generator": {"type": "noise", "noise_type": "white"},
            "volume": 0.25,
            "effects": [{"type": "low_pass", "cutoff": 2_000}],
        }
    )

    sound = SoundDefinition.model_validate(config)

    assert len(sound.layers) == 2
    assert sound.layers[1].generator.type == "noise"
    assert sound.layers[1].envelope.sustain == pytest.approx(1.0)
    assert isinstance(sound.layers[1].effects[0], LowPassEffectDefinition)


@pytest.mark.parametrize(
    ("effect_config", "expected_type"),
    [
        ({"type": "gain", "gain": 0.5}, GainEffectDefinition),
        ({"type": "distortion", "drive_db": 12}, DistortionEffectDefinition),
        ({"type": "low_pass", "cutoff": 1_000}, LowPassEffectDefinition),
        ({"type": "high_pass", "cutoff": 120}, HighPassEffectDefinition),
        (
            {"type": "delay", "delay_seconds": 0.2, "feedback": 0.3, "mix": 0.4},
            DelayEffectDefinition,
        ),
        (
            {
                "type": "reverb",
                "room_size": 0.3,
                "damping": 0.4,
                "wet_level": 0.2,
                "dry_level": 0.8,
                "width": 1.0,
            },
            ReverbEffectDefinition,
        ),
    ],
)
def test_effect_discriminated_union_parsing(
    effect_config: dict,
    expected_type: type,
) -> None:
    config = valid_config()
    config["effects"] = [effect_config]

    sound = SoundDefinition.model_validate(config)

    assert isinstance(sound.effects[0], expected_type)


def test_unknown_effect_type_is_rejected() -> None:
    config = valid_config()
    config["effects"] = [{"type": "flanger", "rate": 2.0}]

    with pytest.raises(ValidationError, match="effects"):
        SoundDefinition.model_validate(config)


def test_unknown_fields_are_rejected() -> None:
    config = valid_config()
    config["mystery_setting"] = True

    with pytest.raises(ValidationError, match="mystery_setting"):
        SoundDefinition.model_validate(config)


def test_filter_cutoff_must_be_below_nyquist_frequency() -> None:
    config = valid_config()
    config["sample_rate"] = 44_100
    config["effects"] = [{"type": "low_pass", "cutoff": 22_050}]

    with pytest.raises(ValidationError, match="Nyquist"):
        SoundDefinition.model_validate(config)
