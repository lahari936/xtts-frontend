"""XTTS-v2 voice cloning UI. Upload/record a reference voice, type text, get it spoken in that voice."""
import os
import tempfile

import gradio as gr

os.environ.setdefault("COQUI_TOS_AGREED", "1")

LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
             "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]

_tts = None


def get_tts():
    global _tts
    if _tts is None:
        import torch
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _tts


def clone(reference_audio, text, language):
    if not reference_audio:
        raise gr.Error("Upload or record a reference voice first (6-30 seconds of clean speech).")
    if not text or not text.strip():
        raise gr.Error("Enter some text to speak.")

    out_path = tempfile.mktemp(suffix=".wav", dir=OUT_DIR)
    get_tts().tts_to_file(
        text=text.strip(),
        speaker_wav=reference_audio,
        language=language,
        file_path=out_path,
    )
    return out_path, reference_audio


OUT_DIR = os.path.join(tempfile.gettempdir(), "xtts_out")
os.makedirs(OUT_DIR, exist_ok=True)

with gr.Blocks(title="XTTS Voice Cloning") as demo:
    gr.Markdown(
        "# XTTS Voice Cloning\n"
        "Give a short reference recording, type any text, and hear it in that voice.\n\n"
        "**Only clone voices you have permission to use.**"
    )

    with gr.Row():
        with gr.Column():
            ref = gr.Audio(label="Reference voice (6-30s)", sources=["upload", "microphone"], type="filepath")
            txt = gr.Textbox(label="Text to speak", lines=4, placeholder="Type what the cloned voice should say...")
            lang = gr.Dropdown(LANGUAGES, value="en", label="Language")
            go = gr.Button("Clone voice", variant="primary")
        with gr.Column():
            out_clone = gr.Audio(label="Cloned voice (download via the ⤓ button)", type="filepath", interactive=False)
            out_ref = gr.Audio(label="Original reference (download via the ⤓ button)", type="filepath", interactive=False)

    go.click(clone, [ref, txt, lang], [out_clone, out_ref])

if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("GRADIO_SHARE") == "1",
    )
