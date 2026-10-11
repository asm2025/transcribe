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

## Usage

```
usage: app.py [-h] [-m MODEL] [--gpu] files [files ...]
```

```bash
# CPU (default), base model
uv run app.py video.mp4

# GPU, small model, single file
uv run app.py -m small --gpu "L:\Shared\recording.mp4"

# GPU, large model, multiple files
uv run app.py -m large-v3 --gpu clip1.mp4 clip2.mkv /abs/path/audio.mp3
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `files` | *(required)* | One or more audio/video file paths (relative or absolute). |
| `-m`, `--model` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`. Larger models are more accurate but slower. |
| `--gpu` | *(off)* | Use NVIDIA GPU (CUDA). Omit to run on CPU. |

## Output

Each input file gets a `.txt` transcript saved in the same directory, with the same base name:

```
L:\Shared\recording.mp4  →  L:\Shared\recording.txt
clips\intro.mkv           →  clips\intro.txt
```

If any file fails or cannot be found, it is skipped and the script exits with a non-zero code. All other files are still processed.

## Notes

- The model is downloaded automatically from Hugging Face on first use and cached in `~/.cache/huggingface/hub`.
- On Windows, Hugging Face caching works without symlinks (a warning is shown but can be ignored or silenced by setting `HF_HUB_DISABLE_SYMLINKS_WARNING=1`).
- Voice-activity detection (`vad_filter`) is enabled by default to skip silence/background noise.
- GPU compute type is `float16`; CPU compute type is `float32`.
