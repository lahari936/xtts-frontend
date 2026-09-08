# XTTS Voice Cloning UI

Gradio app with two tabs:

- **Clone a voice** - clone a voice from a short reference clip using Coqui
  XTTS-v2. Both the generated audio and the reference audio are downloadable,
  and the clone is scored by the detector automatically.
- **Detect AI voice** - run any recording, real or cloned, through the AASIST
  spoof detector in `artifacts/aasist_xtts.onnx`.

## Cloning fidelity and accent

XTTS-v2 is zero-shot. It copies timbre well but rebuilds accent from the accents
in its training data, so a strong regional accent drifts. Two settings matter,
measured by cosine similarity between the XTTS speaker embedding of the
reference set and of the output, three runs each, on a 3-clip Indian-English
speaker:

| configuration | refs | speaker similarity |
|---|---|---|
| single clip, stock conditioning | 1 | 0.530 |
| several clips of the same speaker | 3 | 0.604 |
| several clips + long conditioning | 3 | 0.649 |
| the above at temperature 0.65 | 3 | 0.598 |

So: give it several clips totalling 30 seconds or more, and leave temperature
alone - lowering it did not help. `gpt_cond_len` and `max_ref_len` are set to 30
on `tts_model.config` in `get_tts()`, because `XTTS.synthesize()` re-reads them
from the config after applying kwargs, which means passing them to
`tts_to_file()` has no effect. The stock 12 s / 10 s caps discard most of a long
reference.

This narrows the gap; it does not close it. An accent-exact clone needs the GPT
layer fine-tuned on the target speaker, which is a training job rather than an
inference setting. Matching the language dropdown to the speaker's actual
language moves accent more than any slider in the UI.

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
