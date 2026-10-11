# transcribe

Batch-transcribe audio/video files to text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

## Setup

```bash
cd transcribe
uv sync
```

This installs all dependencies from `uv.lock`, including faster-whisper and the CUDA libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`).

### GPU support (optional)

The CUDA libraries are already installed by `uv sync`; to use them, pass `--gpu`.

> **Note:** Requires an NVIDIA driver with CUDA 12 support (`nvidia-smi` should report "CUDA Version: 12.x" or higher).
> On Windows, the script automatically adds the pip-installed NVIDIA DLL directories to the DLL search path,
> so no manual PATH changes are needed.

### Speaker diarization (optional)

`--diarize` labels each line with its speaker (`Speaker 1`, `Speaker 2`, ...) using
[pyannote.audio](https://github.com/pyannote/pyannote-audio). The library is installed by `uv sync`, but the
model is gated on Hugging Face, so a one-time setup is needed:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Open [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   and accept the user conditions.
3. Create an access token (type **Read**) at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Provide the token, either per run with `--hf-token hf_xxx`, or once via the `HF_TOKEN` environment variable:

   ```powershell
   # PowerShell, current session
   $env:HF_TOKEN = "hf_xxx"
   # PowerShell, permanent (new terminals)
   setx HF_TOKEN "hf_xxx"
   ```

   ```bash
   # bash
   export HF_TOKEN=hf_xxx
   ```

The model is downloaded on first use and cached locally; after that, diarization runs offline.

> **Note:** On Windows, `pyproject.toml` pulls the CUDA 12.6 build of `torch` from the PyTorch index (the PyPI
> build is CPU-only), so `--gpu` also runs diarization on the GPU. Without `--gpu`, diarization runs on the CPU
> and can take a long time on long recordings; progress bars show each stage.

## Usage

```
usage: app.py [-h] [-m MODEL] [--gpu] [--diarize] [--speakers SPEAKERS]
              [--min-speakers MIN_SPEAKERS] [--max-speakers MAX_SPEAKERS]
              [--hf-token HF_TOKEN] files [files ...]
```

```bash
# CPU (default), base model
uv run app.py video.mp4

# GPU, small model, single file
uv run app.py -m small --gpu "L:\Shared\recording.mp4"

# GPU, large model, multiple files
uv run app.py -m large-v3 --gpu clip1.mp4 clip2.mkv /abs/path/audio.mp3

# Label speakers (token from HF_TOKEN)
uv run app.py -m small --gpu --diarize meeting.mp4

# Label speakers, known to be exactly 2 (more accurate)
uv run app.py --diarize --speakers 2 interview.mp3

# Label speakers, somewhere between 2 and 5, token passed inline
uv run app.py --diarize --min-speakers 2 --max-speakers 5 --hf-token hf_xxx call.wav
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `files` | *(required)* | One or more audio/video file paths (relative or absolute). |
| `-m`, `--model` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`. Larger models are more accurate but slower. |
| `--gpu` | *(off)* | Use NVIDIA GPU (CUDA). Omit to run on CPU. |
| `--diarize` | *(off)* | Label each line with its speaker. Requires a Hugging Face token (see [Speaker diarization](#speaker-diarization-optional)). |
| `--speakers` | *(auto)* | Exact number of speakers, if known. Strongly improves accuracy. `--diarize` only. |
| `--min-speakers` | *(auto)* | Lower bound on the number of speakers. `--diarize` only. |
| `--max-speakers` | *(auto)* | Upper bound on the number of speakers. `--diarize` only. |
| `--hf-token` | `HF_TOKEN` env var | Hugging Face access token for the diarization model. |

## Output

Each input file gets a `.txt` transcript saved in the same directory, with the same base name:

```
L:\Shared\recording.mp4  →  L:\Shared\recording.txt
clips\intro.mkv           →  clips\intro.txt
```

Each line is `[start -> end] text`. With `--diarize`, the speaker is added and lines are split wherever the
speaker changes mid-sentence:

```
[0.00s -> 3.42s] Speaker 1: Thanks for joining, let's get started.
[3.42s -> 5.10s] Speaker 2: Sure, sounds good.
```

Speakers are numbered in the order they first speak. They are anonymous: the same person may get a different
number in another file. Overlapping speech is attributed to a single speaker, and very similar voices can be
merged, so check the output on important recordings. `Speaker ?` means no speaker was detected for that line.

If any file fails or cannot be found, it is skipped and the script exits with a non-zero code. All other files are still processed.

## Notes

- The model is downloaded automatically from Hugging Face on first use and cached in `~/.cache/huggingface/hub`.
- On Windows, Hugging Face caching works without symlinks (a warning is shown but can be ignored or silenced by setting `HF_HUB_DISABLE_SYMLINKS_WARNING=1`).
- Voice-activity detection (`vad_filter`) is enabled by default to skip silence/background noise.
- GPU compute type is `float16`; CPU compute type is `float32`.
