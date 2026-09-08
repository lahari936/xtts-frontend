import os, sys, math, wave, struct, tempfile
os.environ["COQUI_TOS_AGREED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ref = os.path.join(tempfile.gettempdir(), "smoke_ref.wav")
sr = 22050
with wave.open(ref, "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    frames = []
    for i in range(sr * 8):
        t = i / sr
        # crude voiced-ish signal: 120 Hz fundamental + harmonics, amplitude modulated
        v = sum(math.sin(2 * math.pi * 120 * k * t) / k for k in (1, 2, 3, 4))
        v *= 0.3 * (1 + math.sin(2 * math.pi * 3 * t))
        frames.append(struct.pack("<h", int(max(-1, min(1, v)) * 20000)))
    w.writeframes(b"".join(frames))
print("ref written", os.path.getsize(ref))

import app
out, echoed = app.clone(ref, "Hello, this is a smoke test of voice cloning.", "en")
print("OUT", out, os.path.getsize(out))
print("ECHO", echoed)
assert os.path.getsize(out) > 10000, "output audio suspiciously small"
print("SMOKE OK")
