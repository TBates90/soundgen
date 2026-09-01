# Procedural Game SFX PoC — Agent Task Plan

## Architecture

The PoC should keep sound definition, synthesis, DSP effects, orchestration, and output rendering separated.

```text
JSON config
    ↓
Pydantic validation
    ↓
SoundDefinition
    ↓
Synthesizer
    ├── Oscillators
    ├── Noise generators
    ├── Envelopes
    └── Modulators
    ↓
AudioBuffer
    ↓
EffectChain
    ├── Filters
    ├── Distortion
    ├── Delay
    └── Reverb
    ↓
Renderer
    ↓
.wav
```

## Task 1 — Bootstrap the Python project

Create a clean Python project for a procedural video-game sound-effect generator.

Use Python 3.12+, NumPy, SciPy, Spotify Pedalboard, Pydantic v2, pytest, and soundfile.

Target structure:

```text
procedural-sfx/
├── pyproject.toml
├── README.md
├── src/
│   └── procedural_sfx/
│       ├── __init__.py
│       ├── models/
│       │   └── __init__.py
│       ├── synthesis/
│       │   └── __init__.py
│       ├── effects/
│       │   └── __init__.py
│       ├── rendering/
│       │   └── __init__.py
│       └── services/
│           └── __init__.py
├── examples/
│   └── configs/
└── tests/
```

Acceptance criteria:

- Package installs successfully.
- `pytest` executes successfully.
- `import procedural_sfx` works.
- Dependencies are declared in `pyproject.toml`.
- Source code uses type hints.
- No application logic exists outside `src/procedural_sfx`.

## Task 2 — Define the JSON configuration schema

Implement Pydantic v2 models for `SoundDefinition`, layers, generators, ADSR envelopes, and effects.

Support generator types: `sine`, `square`, `saw`, `triangle`, and `noise`.

Support effect types: `gain`, `distortion`, `low_pass`, `high_pass`, `delay`, and `reverb`.

Use discriminated unions based on `type`, validate sensible ranges, and keep the models independent of NumPy and Pedalboard.

Acceptance criteria: a JSON string can be converted into a strongly typed `SoundDefinition`.

## Task 3 — Implement the audio buffer abstraction

Create an `AudioBuffer` domain object containing NumPy `float32` samples and a sample rate. Support duration, peak, normalization, gain, mixing, and clipping. Keep file IO out of this class.

Acceptance criteria: `AudioBuffer` becomes the standard interchange format for synthesis and effects code.

## Task 4 — Implement waveform generators

Create a generator protocol and implementations for sine, square, saw, triangle, and white noise. Add a factory from generator config to implementation. Mono audio should be normalized floating point.

Acceptance criteria: every generator defined in the JSON model can produce an `AudioBuffer`.

## Task 5 — Implement amplitude envelopes

Create an envelope protocol and ADSR implementation. Attack, decay, and release are expressed in seconds; sustain is an amplitude ratio. Fit stages proportionally when their total duration exceeds the sound duration. Do not mutate input buffers.

Acceptance criteria: any generated waveform can be shaped by the JSON envelope configuration.

## Task 6 — Add frequency automation

Keep support for constant frequencies and add frequency sweeps with `start`, `end`, and `curve`. Support `linear` and `exponential` curves and share sweep logic across oscillator implementations.

Acceptance criteria: JSON can describe a classic laser sweep such as 1200 Hz to 120 Hz over 300 ms.

## Task 7 — Implement the effect abstraction

Create an `AudioEffect` protocol and `EffectFactory`. Implement gain, low-pass, high-pass, distortion, delay, and reverb, using Spotify Pedalboard where practical. Pedalboard types must remain inside the effects package.

Acceptance criteria: effects can be instantiated entirely from JSON configuration.

## Task 8 — Implement sound layers

Create a `LayerRenderer` with processing order:

```text
Generator
→ Envelope
→ Layer volume
→ Layer effects
```

Keep orchestration outside generators and effects.

Acceptance criteria: one JSON layer renders independently into an `AudioBuffer`.

## Task 9 — Implement multi-layer sound rendering

Create a `SoundRenderer` that renders every layer, mixes them sample-for-sample, applies master effects, applies master volume, and prevents hard clipping. Avoid implicit normalization that changes relative layer loudness.

Acceptance criteria: a complete `SoundDefinition` renders into one `AudioBuffer`.

## Task 10 — Implement WAV output

Add a separate output component using `soundfile`. Write mono 16-bit PCM WAV, create parent directories automatically, and keep file IO separate from synthesis.

Acceptance criteria: an `AudioBuffer` exports as a standard WAV suitable for game engines.

## Task 11 — Add the high-level JSON service

Create `SoundEffectService` with APIs for rendering a JSON config or an already parsed definition. The service should orchestrate validation, rendering, and optional WAV output without containing synthesis logic.

Acceptance criteria:

```python
SoundEffectService().render_json(
    json_config,
    output_path="output/test.wav",
)
```

produces a valid sound file.

## Task 12 — Add example sound presets

Add JSON presets for laser, explosion, pickup, and hit, plus an example renderer script.

Acceptance criteria: all four presets render successfully and sound recognizably different.

## Task 13 — Add parameter modulation architecture

Introduce a reusable `ParameterCurve` abstraction supporting constant, linear, and exponential values. Refactor frequency automation to use it so future parameters can share the same mechanism.

Acceptance criteria: frequency sweeps use the generic parameter-curve system rather than oscillator-specific interpolation.

## Task 14 — Add deterministic procedural variation

Add an optional seed and parameter randomization. Resolve randomized values before rendering so the renderer receives a deterministic resolved definition.

Suggested flow:

```text
SoundDefinition
       ↓
ParameterResolver
       ↓
ResolvedSoundDefinition
       ↓
SoundRenderer
```

Acceptance criteria: identical config plus identical seed produces identical samples, while changing the seed changes the output.

## Task 15 — Add command-line interface

Add commands such as:

```bash
sfx render laser.json
sfx render laser.json --output laser.wav
sfx validate laser.json
sfx render-all examples/configs
```

Prefer `argparse` unless another dependency has a clear benefit.

Acceptance criteria: developers can validate and render effects without writing Python code.

## Task 16 — Architectural review and refactor

Review the implementation for DSP leaking into config models, Pedalboard leaking outside effects, file IO leaking into synthesis, duplicated waveform or automation logic, large orchestration classes, mutable shared arrays, circular imports, poor errors, missing typing, unnecessary abstractions, and test gaps.

Update the README with an architecture section and Mermaid diagram.

Acceptance criteria: configuration, domain audio representation, synthesis, DSP effects, orchestration, and output are clearly separated and all tests pass.

## Task 17 — Create a richer demonstration sound

Create `examples/configs/magic_explosion.json` composed entirely through the public JSON model with multiple layers, filtering/distortion, delay, and reverb. Do not add preset-specific Python code.

Acceptance criteria: the preset demonstrates that sophisticated effects can be composed entirely through configuration.

## PoC completion criteria

The PoC is complete when this workflow works:

```python
from procedural_sfx import SoundEffectService

service = SoundEffectService()
service.render_json(
    config,
    output_path="output/explosion.wav",
)
```

The recommended implementation sequence is Tasks 1–12 first, then Tasks 13–17 as a second iteration. Avoid introducing a generic node/graph DSP language during the PoC; `Sound → Layers → Generator → Envelope → Effects` provides enough flexibility while preserving a clean migration path later.
