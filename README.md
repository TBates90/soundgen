# procedural-sfx

Proof of concept for procedurally generating video-game sound effects from JSON configuration using NumPy for synthesis and Spotify Pedalboard for DSP effects.

## Development

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

The initial package intentionally contains no synthesis implementation yet. Task 1 establishes packaging, dependencies, source layout, and test infrastructure only.
