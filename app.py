"""XTTS-v2 voice cloning UI with AASIST spoof detection.

Clone a voice from a short reference clip, and run any clip through the
AASIST detector to see whether it reads as human or AI-generated.
"""
import json
import os
import tempfile

import gradio as gr
import numpy as np

os.environ.setdefault("COQUI_TOS_AGREED", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("DETECTOR_MODEL", os.path.join(HERE, "artifacts", "aasist_xtts.onnx"))
CALIBRATION_PATH = os.environ.get(
    "DETECTOR_CALIBRATION", os.path.join(HERE, "artifacts", "calibration_xtts.json")
)

DETECTOR_SR = 16000
DETECTOR_SAMPLES = 64600  # 4.04 s, the fixed input width the model was exported with

LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
             "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]

OUT_DIR = os.path.join(tempfile.gettempdir(), "xtts_out")
os.makedirs(OUT_DIR, exist_ok=True)

_tts = None
_detector = None

with open(CALIBRATION_PATH) as f:
    CALIBRATION = json.load(f)
HUMAN_EDGE = CALIBRATION["human_edge"]
SPOOF_EDGE = CALIBRATION["spoof_edge"]


def get_tts():
    global _tts
    if _tts is None:
        import torch
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        # XTTS.synthesize() overwrites these from the config *after* applying any
        # kwargs, so passing them to tts_to_file() does nothing - they have to be
        # set here. The stock 12s/10s caps throw away most of a long reference.
        cfg = _tts.synthesizer.tts_model.config
        cfg.gpt_cond_len = 30
        cfg.gpt_cond_chunk_len = 6
        cfg.max_ref_len = 30
    return _tts


def get_detector():
    global _detector
    if _detector is None:
        import onnxruntime as ort
        _detector = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _detector


def spoof_probability(audio_path):
    """Mean spoof probability over consecutive 4.04 s windows of the clip."""
    import librosa

    wav, _ = librosa.load(audio_path, sr=DETECTOR_SR, mono=True)
    if wav.size == 0:
        raise gr.Error("That audio file is empty.")
    if wav.size < DETECTOR_SAMPLES:
        # AASIST convention: tile the clip up to the fixed input width.
        wav = np.tile(wav, int(np.ceil(DETECTOR_SAMPLES / wav.size)))

    session = get_detector()
    probs = []
    for start in range(0, wav.size - DETECTOR_SAMPLES + 1, DETECTOR_SAMPLES):
        window = wav[start:start + DETECTOR_SAMPLES].astype(np.float32)[None, :]
        logits = session.run(["logits"], {"audio": window})[0][0]
        exp = np.exp(logits - logits.max())
        probs.append(float((exp / exp.sum())[0]))  # index 0 is the spoof class
    return sum(probs) / len(probs), len(probs)


def detect(audio_path):
    if not audio_path:
        raise gr.Error("Upload or record some audio to check first.")

    spoof, windows = spoof_probability(audio_path)

    if spoof >= SPOOF_EDGE:
        verdict = "AI-GENERATED"
    elif spoof < HUMAN_EDGE:
        verdict = "HUMAN"
    else:
        verdict = "UNCERTAIN"

    detail = (
        f"**{verdict}** — spoof probability {spoof:.3f} over {windows} window(s) of 4.04 s.\n\n"
        f"Thresholds: human below {HUMAN_EDGE:.3f}, AI at or above {SPOOF_EDGE:.3f}, "
        f"anything between is declined as uncertain.\n\n"
        f"Model `{CALIBRATION['model_version']}` — trained on {CALIBRATION['domain']}. "
        f"On its validation split it accepted {CALIBRATION['genuine_accept_rate']:.1%} of genuine "
        f"speech and falsely flagged {CALIBRATION['measured_fpr']:.1%} of it."
    )
    return {"AI-generated": spoof, "Human": 1.0 - spoof}, detail


def clone(reference_audio, extra_clips, text, language, temperature):
    if not reference_audio:
        raise gr.Error("Upload or record a reference voice first (6-30 seconds of clean speech).")
    if not text or not text.strip():
        raise gr.Error("Enter some text to speak.")

    # XTTS averages the speaker embedding over every clip it is given, which is
    # the single biggest lever on how closely the clone tracks the original.
    references = [reference_audio] + [f if isinstance(f, str) else f.name for f in (extra_clips or [])]

    tts = get_tts()
    tts.synthesizer.tts_model.config.temperature = float(temperature)

    out_path = tempfile.mktemp(suffix=".wav", dir=OUT_DIR)
    tts.tts_to_file(
        text=text.strip(),
        speaker_wav=references,
        language=language,
        file_path=out_path,
    )
    scores, detail = detect(out_path)
    return out_path, reference_audio, scores, detail


with gr.Blocks(title="XTTS Voice Cloning") as demo:
    gr.Markdown(
        "# XTTS Voice Cloning & Detection\n"
        "Clone a voice from a short reference clip, then check any recording for AI generation.\n\n"
        "**Only clone voices you have permission to use.**"
    )

    with gr.Tab("Clone a voice"):
        with gr.Row():
            with gr.Column():
                ref = gr.Audio(label="Reference voice (6-30s)", sources=["upload", "microphone"], type="filepath")
                extra = gr.File(
                    label="More clips of the same speaker (optional, but strongly recommended)",
                    file_count="multiple",
                    file_types=["audio"],
                )
                txt = gr.Textbox(label="Text to speak", lines=4, placeholder="Type what the cloned voice should say...")
                lang = gr.Dropdown(LANGUAGES, value="en", label="Language")
                temp = gr.Slider(0.5, 1.0, value=0.85, step=0.05, label="Temperature (lower is steadier, not necessarily closer)")
                go = gr.Button("Clone voice", variant="primary")
                gr.Markdown(
                    "**On accent.** XTTS-v2 is zero-shot: it copies timbre well but rebuilds accent "
                    "from the accents it was trained on, so a strong regional accent will drift. "
                    "Feeding it 30 seconds or more across several clips measurably narrows the gap "
                    "(speaker similarity 0.53 to 0.65 on a 3-clip Indian-English speaker), but it "
                    "will not close it. Matching the language dropdown to the speaker's language "
                    "matters more than any slider here. An exact accent-level clone needs the model "
                    "fine-tuned on that speaker, not inference tuning."
                )
            with gr.Column():
                out_clone = gr.Audio(label="Cloned voice (download via the ⤓ button)", type="filepath", interactive=False)
                out_ref = gr.Audio(label="Original reference (download via the ⤓ button)", type="filepath", interactive=False)
                clone_scores = gr.Label(label="Detector verdict on the clone", num_top_classes=2)
                clone_detail = gr.Markdown()

    with gr.Tab("Detect AI voice"):
        gr.Markdown(
            "Pass any recording — a real person or a clone — through the AASIST detector.\n"
            "Clips longer than 4.04 seconds are scored in windows and averaged."
        )
        with gr.Row():
            with gr.Column():
                probe = gr.Audio(label="Audio to check", sources=["upload", "microphone"], type="filepath")
                check = gr.Button("Check this audio", variant="primary")
            with gr.Column():
                probe_scores = gr.Label(label="Verdict", num_top_classes=2)
                probe_detail = gr.Markdown()

    go.click(clone, [ref, extra, txt, lang, temp], [out_clone, out_ref, clone_scores, clone_detail])
    check.click(detect, [probe], [probe_scores, probe_detail])

if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("GRADIO_SHARE") == "1",
    )
