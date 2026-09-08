# XTTS Voice Cloning UI

Gradio app with two tabs:

- **Clone a voice** - clone a voice from a short reference clip using Coqui
  XTTS-v2. Both the generated audio and the reference audio are downloadable,
  and the clone is scored by the detector automatically.
- **Detect AI voice** - run any recording, real or cloned, through the AASIST
  spoof detector in `artifacts/aasist_xtts.onnx`.

## Detection

The detector takes 16 kHz mono audio in fixed 64600-sample (4.04 s) windows;
longer clips are scored window by window and averaged, shorter ones are tiled
up to one window. Spoof probability is `softmax(logits)[0]`.

Thresholds come from `artifacts/calibration_xtts.json` and are three-way:
below `human_edge` (0.777) is HUMAN, at or above `spoof_edge` (0.924) is
AI-GENERATED, and the band between is declined as UNCERTAIN rather than
guessed. That gap is deliberate - the calibration trades a 20% miss rate for a
2.9% rate of falsely accusing real speech.

Measured on the 80 labelled clips in the source project: genuine clips score
0.006 mean (max 0.030 Hindi, 0.926 on one English outlier), XTTS clips score
0.986 mean (min 0.923). The model is an XTTS specialist trained on Hindi and
Indian English - unseen generators are caught far less reliably.

Override the paths with `DETECTOR_MODEL` and `DETECTOR_CALIBRATION`.

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
