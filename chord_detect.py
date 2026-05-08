"""
chord_detect.py — madmom chord detection sidecar.

Runs in the Python 3.8 sidecar venv (venv-chords). madmom doesn't build on
Python 3.14, so chord detection is isolated here and called from the main
pipeline via subprocess.

Usage:
    python chord_detect.py <audio_file>

Stdout: JSON list of {chord, start, end} segments. "N" = no chord (silence/break).
Stderr: progress / log messages (consumed by the main pipeline's logger).

Labels are normalized to chord-scribe's existing vocabulary:
    C:maj -> C
    A:min -> Am
    N     -> N
"""

# Required for `list[dict]` style annotations on Python 3.8.
from __future__ import annotations

import json
import sys


# madmom emits root names with sharps; keep them so we match _NOTES in pipeline.py
def _normalize_label(label: str) -> str:
    """Map madmom's `<root>:<quality>` label to chord-scribe's `<root>[m]` format."""
    if not label or label == "N":
        return "N"
    if ":" not in label:
        return label  # already in target form

    root, quality = label.split(":", 1)
    # Standardize accidentals: madmom uses 'b' (e.g. 'Db') — convert to sharps
    flat_to_sharp = {
        "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    }
    root = flat_to_sharp.get(root, root)

    if quality == "maj":
        return root
    if quality == "min":
        return f"{root}m"
    # madmom's default CRF model only emits maj/min/N; anything else
    # (defensive for future model swaps) falls back to root-only major.
    return root


def detect(audio_path: str) -> list[dict]:
    print(f"[chord_detect] loading madmom processors...", file=sys.stderr, flush=True)
    from madmom.features.chords import (
        CNNChordFeatureProcessor,
        CRFChordRecognitionProcessor,
    )

    feature_proc = CNNChordFeatureProcessor()
    decode_proc = CRFChordRecognitionProcessor()

    print(f"[chord_detect] running CNN feature extraction on {audio_path}...",
          file=sys.stderr, flush=True)
    features = feature_proc(audio_path)

    print(f"[chord_detect] CRF decoding...", file=sys.stderr, flush=True)
    raw_segments = decode_proc(features)
    # raw_segments is a numpy structured array of (start_time, end_time, label)

    segments = []
    for start, end, label in raw_segments:
        segments.append({
            "chord": _normalize_label(str(label)),
            "start": float(start),
            "end": float(end),
        })

    real = sum(1 for s in segments if s["chord"] != "N")
    print(f"[chord_detect] detected {real} chord segments "
          f"({len(segments)} total inc. silence)", file=sys.stderr, flush=True)
    return segments


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python chord_detect.py <audio_file>", file=sys.stderr)
        return 2
    audio_path = sys.argv[1]
    segments = detect(audio_path)
    json.dump(segments, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
