#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


def _register_nvidia_dlls():
    """On Windows, make DLLs from pip-installed nvidia-* packages (cuBLAS, cuDNN)
    discoverable. Those wheels put DLLs in site-packages/nvidia/<lib>/bin, which
    is not on the DLL search path by default."""
    if os.name != "nt":
        return
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_register_nvidia_dlls()

from faster_whisper import WhisperModel  # noqa: E402

# Compatibility fix: PyAV >= 19 removed the 'metadata_errors' kwarg that
# faster-whisper passes internally. Patch it out so both old and new PyAV work.
import av as _av
_av_open_orig = _av.open
def _av_open_patched(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _av_open_orig(*args, **kwargs)
_av.open = _av_open_patched

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def parse_args():
    p = argparse.ArgumentParser(description="Transcribe audio/video files with faster-whisper.")
    p.add_argument("files", nargs="+", help="One or more file paths (relative or absolute). Positional - pass paths "
                        "directly, no 'files' keyword needed. Example: app.py -m small --gpu \"C:\\path\\video.mp4\"")
    p.add_argument("-m", "--model", default="base",
                   help=f"Model size (default: base). Examples: {', '.join(MODEL_SIZES)}")
    p.add_argument("--gpu", action="store_true", help="Use GPU (CUDA). Default is CPU.")
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if args.gpu else "cpu"
    compute_type = "float16" if args.gpu else "float32"

    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as e:
        sys.exit(f"Failed to load model '{args.model}' on {device}: {e}")

    failed = 0
    for f in args.files:
        path = Path(f).expanduser().resolve()
        if not path.is_file():
            print(f"[skip] Not a file: {path}", file=sys.stderr)
            failed += 1
            continue

        out_path = path.with_suffix(".txt")  # same name, new extension
        print(f"\nTranscribing: {path}")

        try:
            segments, info = model.transcribe(str(path), beam_size=5, vad_filter=True)
            print(f"Detected language '{info.language}' "
                  f"(probability {info.language_probability:.2f})")

            with open(out_path, "w", encoding="utf-8") as out:
                for seg in segments:
                    line = f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text.strip()}"
                    print(line)
                    out.write(line + "\n")
        except Exception as e:
            print(f"[error] {path}: {e}", file=sys.stderr)
            failed += 1
            continue

        print(f"Saved to {out_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()