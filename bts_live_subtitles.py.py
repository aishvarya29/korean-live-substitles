"""
Live Korean -> English subtitles for Windows.

Captures whatever audio your laptop is playing (WASAPI loopback, no VB-Cable needed),
translates it with Whisper, and shows subtitles in an always-on-top window.

Setup (in Command Prompt / PowerShell):
    pip install faster-whisper PyAudioWPatch numpy
Run:
    python bts_live_subtitles.py

Play the live in your browser, with sound through your normal speakers/headphones.
"""

import queue
import threading
import tkinter as tk

import numpy as np
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel

# ---- Settings you can change -------------------------------------------------
MODEL_SIZE = "base"      # tiny / base / small / medium. Bigger = better but slower.
DEVICE = "cpu"            # use "cuda" if you have an NVIDIA GPU
COMPUTE_TYPE = "int8"     # use "float16" with cuda
CHUNK_SECONDS = 3         # shorter = faster subtitles, but more cut-off words
SILENCE_LEVEL = 0.005     # chunks quieter than this are skipped
FONT_SIZE = 22
# -----------------------------------------------------------------------------

# Phrases Whisper often invents during silence or music. Matching lines are hidden.
HALLUCINATIONS = [
    "thank you for watching",
    "thanks for watching",
    "thank you so much for watching",
    "please subscribe",
    "subscribe to my channel",
    "see you in the next video",
    "see you next time",
    "bye-bye",
    "subtitles by",
    "amara.org",
    "translated by",
    "i'll see you next time",
]


def is_hallucination(text: str) -> bool:
    t = text.lower().strip(" .!?,")
    if len(t) < 2:
        return True
    if any(h in t for h in HALLUCINATIONS):
        return True
    if t in ("you", "thank you", "thanks", "okay", "oh", "uh", "um"):
        return True
    return False

TARGET_SR = 16000
audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=10)
text_q: "queue.Queue[str]" = queue.Queue()


def find_loopback_device(p):
    """Find the loopback twin of the default speakers."""
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if speakers.get("isLoopbackDevice"):
        return speakers
    for dev in p.get_loopback_device_info_generator():
        if speakers["name"] in dev["name"]:
            return dev
    raise RuntimeError("Could not find a loopback device for your speakers.")


def capture_loop():
    p = pyaudio.PyAudio()
    dev = find_loopback_device(p)
    channels = dev["maxInputChannels"]
    rate = int(dev["defaultSampleRate"])
    frames_needed = rate * CHUNK_SECONDS

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=dev["index"],
        frames_per_buffer=1024,
    )

    buf = []
    got = 0
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        buf.append(data)
        got += 1024
        if got >= frames_needed:
            raw = np.frombuffer(b"".join(buf), dtype=np.int16).astype(np.float32) / 32768.0
            buf, got = [], 0
            mono = raw.reshape(-1, channels).mean(axis=1)
            # resample to 16 kHz for Whisper
            n_out = int(len(mono) * TARGET_SR / rate)
            audio = np.interp(
                np.linspace(0, len(mono) - 1, n_out), np.arange(len(mono)), mono
            ).astype(np.float32)
            try:
                audio_q.put_nowait(audio)
            except queue.Full:
                pass  # drop a chunk rather than fall behind


def transcribe_loop():
    text_q.put("Loading model... (first run downloads it, please wait)")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    text_q.put("Ready. Play the live now.")
    while True:
        audio = audio_q.get()
        if np.sqrt(np.mean(audio**2)) < SILENCE_LEVEL:
            continue
        segments, _ = model.transcribe(
            audio,
            task="translate",   # Korean speech -> English text
            language="ko",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,  # stops repeated/looping text
            no_speech_threshold=0.6,
        )
        parts = []
        for s in segments:
            # drop segments Whisper itself thinks are not speech or are low confidence
            if s.no_speech_prob > 0.6 or s.avg_logprob < -1.0:
                continue
            parts.append(s.text.strip())
        text = " ".join(parts).strip()
        if not text or is_hallucination(text):
            continue
        text_q.put(text)


def run_overlay():
    root = tk.Tk()
    root.title("Live Subtitles")
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.88)
    root.configure(bg="black")
    root.geometry("900x130+200+800")

    label = tk.Label(
        root, text="Starting...", fg="white", bg="black",
        font=("Segoe UI", FONT_SIZE, "bold"), wraplength=860, justify="center",
    )
    label.pack(expand=True, fill="both", padx=10, pady=5)

    lines = []

    def poll():
        try:
            while True:
                lines.append(text_q.get_nowait())
                del lines[:-2]  # keep the last 2 lines
                label.config(text="\n".join(lines))
        except queue.Empty:
            pass
        root.after(200, poll)

    # drag the window with the mouse
    def start_move(e):
        root._x, root._y = e.x, e.y

    def do_move(e):
        root.geometry(f"+{root.winfo_x() + e.x - root._x}+{root.winfo_y() + e.y - root._y}")

    label.bind("<Button-1>", start_move)
    label.bind("<B1-Motion>", do_move)

    poll()
    root.mainloop()


if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=transcribe_loop, daemon=True).start()
    run_overlay()