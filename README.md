# XTTS Voice Cloning UI

Gradio app that clones a voice from a short reference clip using Coqui XTTS-v2.
Both the generated audio and the reference audio are downloadable from the UI.

## Requirements

Python 3.9-3.12 (3.11 recommended). XTTS does not run on Python 3.13+.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python app.py
```

Opens on http://localhost:7860. The XTTS-v2 model (~1.8 GB) downloads on the
first generation, not at startup.

## Deploy

- `PORT` sets the port (default 7860).
- `GRADIO_SHARE=1` creates a public gradio.live tunnel.
- Container: `python:3.11-slim`, `apt-get install -y ffmpeg`, then the setup and
  run steps above. GPU optional; CPU works but is roughly 10x slower.

## Notes

XTTS-v2 is licensed under the Coqui Public Model License (non-commercial).
Only clone voices you have permission to use.
