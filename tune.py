"""
Tuning script for chord-scribe pipeline constants.
Caches transcription and chord data after first run, then iterates
over parameter combinations quickly without re-running ML models.

Usage:
  python tune.py <audio_file> [--vocals <vocals_stem>]
"""

import argparse
import json
import pathlib
import textwrap

import pipeline


CACHE_DIR = pathlib.Path("chord_scribe_work")

# ── Parameter sets to try ────────────────────────────────────────────────────
# Each dict overrides the defaults for that iteration.
ITERATIONS = [
    # 1 — baseline: no max-words limit (old behaviour)
    {"LINE_GAP_THRESHOLD": 1.0, "LINE_MAX_WORDS": 999, "CHORD_HOP_SECONDS": 0.5, "CHORD_MIN_DURATION": 1.0,  "CHORD_ENERGY_THRESHOLD": 0.01},
    # 2 — max 10 words, generous gap
    {"LINE_GAP_THRESHOLD": 1.0, "LINE_MAX_WORDS": 10,  "CHORD_HOP_SECONDS": 0.5, "CHORD_MIN_DURATION": 1.0,  "CHORD_ENERGY_THRESHOLD": 0.01},
    # 3 — max 8 words
    {"LINE_GAP_THRESHOLD": 1.0, "LINE_MAX_WORDS": 8,   "CHORD_HOP_SECONDS": 0.5, "CHORD_MIN_DURATION": 1.0,  "CHORD_ENERGY_THRESHOLD": 0.01},
    # 4 — max 6 words (short punchy lines)
    {"LINE_GAP_THRESHOLD": 1.0, "LINE_MAX_WORDS": 6,   "CHORD_HOP_SECONDS": 0.5, "CHORD_MIN_DURATION": 1.0,  "CHORD_ENERGY_THRESHOLD": 0.01},
    # 5 — max 8 words + finer chord hop + lower energy
    {"LINE_GAP_THRESHOLD": 1.0, "LINE_MAX_WORDS": 8,   "CHORD_HOP_SECONDS": 0.25, "CHORD_MIN_DURATION": 0.5, "CHORD_ENERGY_THRESHOLD": 0.005},
    # 6 — max 8 words + very low energy threshold
    {"LINE_GAP_THRESHOLD": 1.0, "LINE_MAX_WORDS": 8,   "CHORD_HOP_SECONDS": 0.25, "CHORD_MIN_DURATION": 0.5, "CHORD_ENERGY_THRESHOLD": 0.001},
    # 7 — max 8 words + tighter gap threshold too
    {"LINE_GAP_THRESHOLD": 0.5, "LINE_MAX_WORDS": 8,   "CHORD_HOP_SECONDS": 0.25, "CHORD_MIN_DURATION": 0.5, "CHORD_ENERGY_THRESHOLD": 0.005},
    # 8 — max 10 words + tighter gap + finer chord
    {"LINE_GAP_THRESHOLD": 0.5, "LINE_MAX_WORDS": 10,  "CHORD_HOP_SECONDS": 0.25, "CHORD_MIN_DURATION": 0.5, "CHORD_ENERGY_THRESHOLD": 0.005},
    # 9 — balanced sweet spot candidate
    {"LINE_GAP_THRESHOLD": 0.8, "LINE_MAX_WORDS": 8,   "CHORD_HOP_SECONDS": 0.25, "CHORD_MIN_DURATION": 0.5, "CHORD_ENERGY_THRESHOLD": 0.003},
    # 10 — max 12 words (longer lines, fewer chord changes)
    {"LINE_GAP_THRESHOLD": 1.0, "LINE_MAX_WORDS": 12,  "CHORD_HOP_SECONDS": 0.25, "CHORD_MIN_DURATION": 1.0, "CHORD_ENERGY_THRESHOLD": 0.005},
]


def load_or_run_transcription(vocals_path: str) -> list[dict]:
    cache_file = CACHE_DIR / "words_cache.json"
    if cache_file.exists():
        print(f"[cache] Loading transcription from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)
    print("[cache] Running transcription (will cache for future iterations)...")
    words = pipeline.transcribe(vocals_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(words, f)
    print(f"[cache] Saved to {cache_file}")
    return words


def load_or_run_chords(audio_path: str) -> list[dict]:
    cache_file = CACHE_DIR / "chords_cache.json"
    if cache_file.exists():
        print(f"[cache] Loading chords from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)
    print("[cache] Running chord detection (will cache for future iterations)...")
    chords = pipeline.detect_chords(audio_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(chords, f)
    print(f"[cache] Saved to {cache_file}")
    return chords


def apply_params(params: dict):
    for key, val in params.items():
        setattr(pipeline, key, val)


def run_iteration(n: int, params: dict, words: list[dict], chords: list[dict]):
    apply_params(params)
    lines = pipeline.group_into_lines(words)
    chopro = pipeline.build_chordpro(lines, chords, title="Golden Fool", artist="Unknown")

    chord_count = chopro.count("[")
    line_count = len([l for l in chopro.splitlines() if l and not l.startswith("{")])

    print(f"\n{'='*70}")
    print(f"Iteration {n:2d} | gap={params['LINE_GAP_THRESHOLD']}s  "
          f"max_words={params['LINE_MAX_WORDS']}  "
          f"hop={params['CHORD_HOP_SECONDS']}s  "
          f"min_dur={params['CHORD_MIN_DURATION']}s  "
          f"energy={params['CHORD_ENERGY_THRESHOLD']}")
    print(f"           | {line_count} lines, {chord_count} chord annotations")
    print(f"{'-'*70}")
    # Show first 8 lines as a preview
    preview_lines = [l for l in chopro.splitlines() if l and not l.startswith("{")][:8]
    for line in preview_lines:
        print(textwrap.fill(line, width=68, subsequent_indent="  "))
    if line_count > 8:
        print(f"  ... ({line_count - 8} more lines)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Original audio file (for chord detection)")
    parser.add_argument("--vocals", default="", help="Path to vocals stem (skips Demucs)")
    args = parser.parse_args()

    audio_path = str(pathlib.Path(args.audio_file).resolve())

    if args.vocals:
        vocals_path = args.vocals
    else:
        default_vocals = CACHE_DIR / f"{pathlib.Path(audio_path).stem}_vocals.wav"
        if default_vocals.exists():
            vocals_path = str(default_vocals)
            print(f"[cache] Found existing vocals stem: {vocals_path}")
        else:
            print("No vocals stem found. Run the full pipeline first or pass --vocals.")
            return

    words = load_or_run_transcription(vocals_path)
    chords = load_or_run_chords(audio_path)

    for i, params in enumerate(ITERATIONS, 1):
        run_iteration(i, params, words, chords)

    print(f"\n{'='*70}")
    print("Done. Adjust ITERATIONS in tune.py to explore further.")



if __name__ == "__main__":
    main()
