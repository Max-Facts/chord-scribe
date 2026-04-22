"""
chord-scribe pipeline
Stages: source separation → transcription → chord detection → align + format
"""

import os
import subprocess
import pathlib
import torch
import autochord
from faster_whisper import WhisperModel

# Tunable constants
LINE_GAP_THRESHOLD = 1.0  # seconds of silence that starts a new lyric line


# ---------------------------------------------------------------------------
# Stage 1 — Source separation (Demucs)
# ---------------------------------------------------------------------------

def separate_vocals(audio_path: str, output_dir: str) -> str:
    """
    Run Demucs htdemucs on audio_path, return path to the vocals stem.
    output_dir is the base folder where Demucs writes its results.
    """
    audio_path = pathlib.Path(audio_path).resolve()
    output_dir = pathlib.Path(output_dir).resolve()

    print("[1/4] Separating vocals (Demucs htdemucs)...")

    subprocess.run(
        [
            "python", "-m", "demucs",
            "--name", "htdemucs",
            "--out", str(output_dir),
            str(audio_path),
        ],
        check=True,
    )

    # Demucs output: <output_dir>/htdemucs/<stem_name>/vocals.wav
    stem_name = audio_path.stem
    vocals_path = output_dir / "htdemucs" / stem_name / "vocals.wav"

    if not vocals_path.exists():
        raise FileNotFoundError(f"Demucs vocals stem not found at {vocals_path}")

    print(f"    Vocals stem: {vocals_path}")
    return str(vocals_path)


# ---------------------------------------------------------------------------
# Stage 2 — Transcription (faster-whisper)
# ---------------------------------------------------------------------------

def transcribe(vocals_path: str) -> list[dict]:
    """
    Transcribe the vocals stem with faster-whisper large-v2.
    Returns a list of word dicts: {word, start, end}
    """
    print("[2/4] Transcribing vocals (faster-whisper large-v2)...")

    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    print(f"    Device: {device} / compute_type: {compute_type}")

    model = WhisperModel("large-v2", device=device, compute_type=compute_type)
    segments, _ = model.transcribe(vocals_path, word_timestamps=True)

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({"word": w.word, "start": w.start, "end": w.end})

    print(f"    Transcribed {len(words)} words.")
    return words


def group_into_lines(words: list[dict]) -> list[dict]:
    """
    Group words into lyric lines based on LINE_GAP_THRESHOLD.
    Returns list of line dicts: {text, start, end}
    """
    if not words:
        return []

    lines = []
    current = [words[0]]

    for word in words[1:]:
        gap = word["start"] - current[-1]["end"]
        if gap >= LINE_GAP_THRESHOLD:
            lines.append(_make_line(current))
            current = [word]
        else:
            current.append(word)

    if current:
        lines.append(_make_line(current))

    return lines


def _make_line(words: list[dict]) -> dict:
    return {
        "text": " ".join(w["word"].strip() for w in words),
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
    }


# ---------------------------------------------------------------------------
# Stage 3 — Chord detection (autochord)
# ---------------------------------------------------------------------------

def detect_chords(audio_path: str) -> list[dict]:
    """
    Detect chords from the original audio file.
    Returns list of dicts: {chord, start, end}
    """
    print("[3/4] Detecting chords (autochord)...")

    raw = autochord.recognize(audio_path)
    chords = [{"chord": c, "start": s, "end": e} for s, e, c in raw]

    real = [c for c in chords if c["chord"] != "N"]
    print(f"    Detected {len(real)} chord segments (excluding silences).")
    return chords


# ---------------------------------------------------------------------------
# Stage 4 — Alignment + ChordPro formatting
# ---------------------------------------------------------------------------

def _chord_at(chords: list[dict], time: float) -> str | None:
    """Return the chord active at `time`, or None if none / 'N'."""
    for c in chords:
        if c["start"] <= time < c["end"]:
            chord = c["chord"]
            return None if chord == "N" else chord
    return None


def build_chordpro(
    lines: list[dict],
    chords: list[dict],
    title: str = "",
    artist: str = "",
) -> str:
    """
    Align chords to lyric lines and produce ChordPro text.
    Chords are only annotated when they change.
    """
    print("[4/4] Aligning chords and formatting ChordPro...")

    out = []

    if title:
        out.append(f"{{title: {title}}}")
    if artist:
        out.append(f"{{artist: {artist}}}")
    if title or artist:
        out.append("")

    prev_chord = None

    for line in lines:
        chord = _chord_at(chords, line["start"])

        if chord and chord != prev_chord:
            # Insert chord marker at the start of the first word
            annotated = f"[{chord}]{line['text']}"
            prev_chord = chord
        else:
            annotated = line["text"]

        out.append(annotated)

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def process(
    audio_path: str,
    output_path: str,
    title: str = "",
    artist: str = "",
    work_dir: str | None = None,
) -> str:
    """
    Run the full pipeline. Returns the ChordPro text.
    Writes output to output_path.
    """
    audio_path = str(pathlib.Path(audio_path).resolve())

    if work_dir is None:
        work_dir = str(pathlib.Path(audio_path).parent / "chord_scribe_work")
    pathlib.Path(work_dir).mkdir(parents=True, exist_ok=True)

    vocals_path = separate_vocals(audio_path, work_dir)
    words = transcribe(vocals_path)
    lines = group_into_lines(words)
    chords = detect_chords(audio_path)
    chopro = build_chordpro(lines, chords, title=title, artist=artist)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(chopro)

    print(f"\nSaved: {output_path}")
    return chopro
