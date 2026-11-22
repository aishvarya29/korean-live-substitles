# Live Korean to English Subtitles (Windows)

A small Python tool that listens to the audio playing on your Windows laptop and shows live English subtitles for Korean speech in an always-on-top window. Made for watching Korean live streams without built-in translated subtitles.

It uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) to translate Korean speech to English on your own computer. Nothing is sent to any server after the model is downloaded.

## Requirements

- Windows 10 or 11
- Python 3.12 or 3.13 (tick "Add python.exe to PATH" when installing)
- About 1 GB free disk space for the speech model

## Install

```
python -m pip install faster-whisper PyAudioWPatch numpy
```

## Run

```
python bts_live_subtitles.py
```

The first run downloads the speech model, which can take several minutes. When the window shows "Ready. Play the live now.", start playing your video in a browser. Sound must come out of your normal speakers or headphones.

Drag the subtitle window with the mouse to place it where you want.

## Settings

Edit the values at the top of `bts_live_subtitles.py`:

| Setting | What it does |
| --- | --- |
| `MODEL_SIZE` | `tiny`, `base`, `small`, `medium`. Bigger is more accurate but slower. |
| `DEVICE` / `COMPUTE_TYPE` | Use `cuda` and `float16` if you have an NVIDIA GPU. |
| `CHUNK_SECONDS` | Shorter gives faster subtitles but more cut-off sentences. |
| `SILENCE_LEVEL` | Audio quieter than this is ignored. |
| `HALLUCINATIONS` | Phrases Whisper tends to invent during silence ("Thank you for watching"). Matching lines are hidden. |

## Known limits

- Expect a delay of about 4 to 8 seconds.
- Singing, loud music and people talking over each other are translated poorly.
- Names and slang are sometimes wrong. Direct Whisper translation is the weakest part. A better approach is to transcribe Korean text and translate it with a stronger translation model.
- Works only with audio played on the same computer.

## Note

For personal use only. Do not record or redistribute copyrighted streams or their subtitles.
