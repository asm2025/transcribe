# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Single-script CLI (`app.py`) that batch-transcribes audio/video files to `.txt` using faster-whisper. Managed with `uv`; Python 3.13 (`.python-version`). No tests, linter, or build step are configured.

## Commands

```bash
uv sync                                   # install deps from uv.lock
uv run app.py video.mp4                   # CPU, base model
uv run app.py -m large-v3 --gpu a.mp4 b.mkv   # GPU, multiple files
uv add <pkg>                              # add dependency (updates pyproject.toml + uv.lock)
```

Models download from Hugging Face on first use (cached in `~/.cache/huggingface/hub`). Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence the Windows symlink warning.

## Architecture notes

`app.py` import order matters — two module-level hacks run before/around the faster-whisper import:

1. `_register_nvidia_dlls()` runs **before** `from faster_whisper import WhisperModel`. On Windows, the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels put DLLs in `site-packages/nvidia/<lib>/bin`, which isn't on the DLL search path; this adds each to `os.add_dll_directory` and `PATH`. Moving the import above this call breaks `--gpu` on Windows.
2. `av.open` is monkey-patched to drop the `metadata_errors` kwarg, because PyAV >= 19 removed it but faster-whisper still passes it.

Behavior to preserve:
- Output goes next to each input: `path.with_suffix(".txt")`, one `[start -> end] text` line per segment, same as stdout.
- `transcribe(..., beam_size=5, vad_filter=True)`; GPU uses `float16`, CPU uses `float32`.
- `--diarize` (pyannote `speaker-diarization-community-1`, needs HF token) imports torch/pyannote lazily, decodes audio via `faster_whisper.decode_audio` (avoids torchcodec/FFmpeg), enables `word_timestamps`, and splits lines on speaker change: `[start -> end] Speaker N: text`. PyPI torch on Windows is CPU-only.
- Missing/failed files are skipped; remaining files still process; exit code is 1 if any failed. Model load failure exits immediately.
