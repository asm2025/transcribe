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
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def parse_args():
    p = argparse.ArgumentParser(description="Transcribe audio/video files with faster-whisper.")
    p.add_argument("files", nargs="+", help="One or more file paths (relative or absolute). Positional - pass paths "
                        "directly, no 'files' keyword needed. Example: app.py -m small --gpu \"C:\\path\\video.mp4\"")
    p.add_argument("-m", "--model", default="base",
                   help=f"Model size (default: base). Examples: {', '.join(MODEL_SIZES)}")
    p.add_argument("--gpu", action="store_true", help="Use GPU (CUDA). Default is CPU.")
    p.add_argument("--diarize", action="store_true",
                   help="Label each line with its speaker (Speaker 1, Speaker 2, ...) using pyannote.audio. "
                        "Requires a Hugging Face token (--hf-token or HF_TOKEN env var).")
    p.add_argument("--speakers", type=int, help="Exact number of speakers, if known (improves --diarize accuracy).")
    p.add_argument("--min-speakers", type=int, help="Minimum number of speakers (--diarize only).")
    p.add_argument("--max-speakers", type=int, help="Maximum number of speakers (--diarize only).")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="Hugging Face access token for the diarization model (default: HF_TOKEN env var).")
    return p.parse_args()


def load_diarization(token, use_gpu):
    """Load the pyannote speaker-diarization pipeline. Imported lazily so torch
    is only loaded when --diarize is used."""
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    if pipeline is None:
        raise RuntimeError("download failed - check your token and that you accepted "
                           "the model terms on huggingface.co")
    if use_gpu:
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        else:
            print("[warn] Installed torch has no CUDA support; diarization runs on CPU.", file=sys.stderr)
    return pipeline


def diarize(pipeline, path, args):
    """Return [(start, end, speaker_label)] turns, labels numbered by first appearance."""
    import torch
    from faster_whisper import decode_audio

    # Decode with faster-whisper's PyAV loader (16 kHz mono) so pyannote doesn't
    # need its own audio backend (torchcodec + FFmpeg DLLs).
    audio = decode_audio(str(path), sampling_rate=16000)
    kwargs = {k: v for k, v in (("num_speakers", args.speakers),
                                ("min_speakers", args.min_speakers),
                                ("max_speakers", args.max_speakers)) if v is not None}
    output = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}, **kwargs)
    # Exclusive diarization has no overlapping turns, which suits transcript alignment.
    annotation = output.exclusive_speaker_diarization

    names = {}
    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        names.setdefault(speaker, f"Speaker {len(names) + 1}")
        turns.append((turn.start, turn.end, names[speaker]))
    return turns


def speaker_at(turns, start, end):
    """Speaker whose turns overlap [start, end] the most; nearest turn if none overlap."""
    if not turns:
        return "Speaker ?"
    overlap = {}
    for t_start, t_end, label in turns:
        o = min(end, t_end) - max(start, t_start)
        if o > 0:
            overlap[label] = overlap.get(label, 0) + o
    if overlap:
        return max(overlap, key=overlap.get)
    mid = (start + end) / 2
    return min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))[2]


def speaker_lines(segments, turns):
    """Yield (start, end, speaker, text), splitting segments at word level where the speaker changes."""
    for seg in segments:
        if not seg.words:
            yield seg.start, seg.end, speaker_at(turns, seg.start, seg.end), seg.text.strip()
            continue
        current = None
        for w in seg.words:
            label = speaker_at(turns, w.start, w.end)
            if current and current[2] == label:
                current[1] = w.end
                current[3] += w.word
            else:
                if current:
                    yield current[0], current[1], current[2], current[3].strip()
                current = [w.start, w.end, label, w.word]
        yield current[0], current[1], current[2], current[3].strip()


def main():
    args = parse_args()

    device = "cuda" if args.gpu else "cpu"
    compute_type = "float16" if args.gpu else "float32"

    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as e:
        sys.exit(f"Failed to load model '{args.model}' on {device}: {e}")

    diarizer = None
    if args.diarize:
        if not args.hf_token:
            sys.exit("--diarize needs a Hugging Face token: pass --hf-token or set HF_TOKEN.")
        try:
            diarizer = load_diarization(args.hf_token, args.gpu)
        except Exception as e:
            sys.exit(f"Failed to load diarization model '{DIARIZATION_MODEL}': {e}")

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
            segments, info = model.transcribe(str(path), beam_size=5, vad_filter=True,
                                              word_timestamps=diarizer is not None)
            print(f"Detected language '{info.language}' "
                  f"(probability {info.language_probability:.2f})")

            if diarizer:
                print("Diarizing speakers...")
                turns = diarize(diarizer, path, args)
                lines = (f"[{start:.2f}s -> {end:.2f}s] {speaker}: {text}"
                         for start, end, speaker, text in speaker_lines(segments, turns))
            else:
                lines = (f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text.strip()}" for seg in segments)

            with open(out_path, "w", encoding="utf-8") as out:
                for line in lines:
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