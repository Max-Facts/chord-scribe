"""
chord-scribe pipeline
Stages: source separation → transcription → chord detection → align + format
"""

import json
import logging
import pathlib
import subprocess
import sys
import numpy as np
import librosa
import soundfile as sf
import torch
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

# Tunable constants
LINE_GAP_THRESHOLD = 0.5  # seconds of silence that starts a new lyric line
LINE_MAX_WORDS = 14       # max words per line before forcing a break (fallback for sung lyrics)


# ---------------------------------------------------------------------------
# Stage 1 — Source separation (Demucs)
# ---------------------------------------------------------------------------

def separate_stems(audio_path: str, output_dir: str) -> tuple[str, str]:
    """
    Run Demucs htdemucs on audio_path.
    Returns (vocals_path, other_path) — both saved as WAV files.
    The 'other' stem (guitar, piano, synths) is used for chord detection.
    Uses soundfile for I/O to avoid torchaudio / torchcodec issues.
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    audio_path = pathlib.Path(audio_path).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vocals_path = output_dir / f"{audio_path.stem}_vocals.wav"
    other_path  = output_dir / f"{audio_path.stem}_other.wav"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Stage 1: separating stems with Demucs htdemucs (device=%s)", device)
    print(f"[1/4] Separating stems (Demucs htdemucs, device={device})...")

    wav_np, sr = sf.read(str(audio_path), always_2d=True)  # (samples, channels)
    wav_np = wav_np.T.astype(np.float32)                   # (channels, samples)
    wav = torch.from_numpy(wav_np).to(device)

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    model = get_model("htdemucs")
    model.to(device)
    model.eval()

    ref = wav.mean(0)
    wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)

    with torch.no_grad():
        sources = apply_model(model, wav_norm[None], progress=True)[0]

    def _save_stem(name: str, path: pathlib.Path):
        idx = model.sources.index(name)
        tensor = sources[idx] * (ref.std() + 1e-8) + ref.mean()
        sf.write(str(path), tensor.cpu().numpy().T, sr)
        log.info("Saved stem '%s' -> %s", name, path)

    _save_stem("vocals", vocals_path)
    _save_stem("other",  other_path)

    print(f"    Vocals: {vocals_path}")
    print(f"    Other (guitar/piano): {other_path}")
    return str(vocals_path), str(other_path)


# ---------------------------------------------------------------------------
# Stage 2 — Transcription (faster-whisper)
# ---------------------------------------------------------------------------

def transcribe(vocals_path: str) -> list[dict]:
    """
    Transcribe the vocals stem with faster-whisper large-v2.
    Returns a list of word dicts: {word, start, end}
    """
    log.info("Stage 2: transcribing %s", vocals_path)
    print("[2/4] Transcribing vocals (faster-whisper large-v2)...")

    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    log.info("Whisper device=%s compute_type=%s", device, compute_type)
    print(f"    Device: {device} / compute_type: {compute_type}")

    model = WhisperModel("large-v2", device=device, compute_type=compute_type)
    segments, _ = model.transcribe(vocals_path, word_timestamps=True)

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({"word": w.word, "start": w.start, "end": w.end})

    log.info("Transcribed %d words", len(words))
    print(f"    Transcribed {len(words)} words.")
    return words


def group_into_lines(words: list[dict]) -> list[dict]:
    """
    Group words into lyric lines.
    Break conditions (in priority order):
    1. Gap >= LINE_GAP_THRESHOLD between words (natural pause)
    2. Previous word ends with punctuation AND line >= LINE_MIN_WORDS (phrase end)
    3. Line has reached LINE_MAX_WORDS (hard fallback for sung lyrics)
    Returns list of line dicts: {text, start, end, words}
    """
    if not words:
        return []

    LINE_MIN_WORDS = 4  # don't break on soft signals until at least this many words

    lines = []
    current = [words[0]]

    for word in words[1:]:
        gap = word["start"] - current[-1]["end"]
        prev_text = current[-1]["word"].strip()
        next_text = word["word"].strip()

        # Hard punctuation ending (no comma — commas are mid-phrase)
        ends_phrase = prev_text and prev_text[-1] in ".!?;"
        # Whisper capitalizes the first word of a new phrase even without punctuation.
        # Skip single-char words (e.g. "I" is always capitalized in English).
        next_capitalized = bool(len(next_text) > 1 and next_text[0].isupper())

        soft_break = (ends_phrase or next_capitalized) and len(current) >= LINE_MIN_WORDS

        if gap >= LINE_GAP_THRESHOLD or soft_break or len(current) >= LINE_MAX_WORDS:
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
# Stage 3 — Chord detection (madmom CNN+CRF, via Python 3.8 sidecar venv)
# ---------------------------------------------------------------------------
#
# madmom doesn't build on Python 3.14, so chord detection runs in a separate
# 3.10 venv (venv-chords/) created by setup-chords-venv.bat. The main pipeline
# shells out to chord_detect.py with the audio path and reads JSON segments
# back from stdout. The CRF in madmom handles segmentation and smoothing
# internally — no onset detection, template matching, min-duration filter,
# or lookahead heuristic needed.

# Chord vocabulary: madmom's pretrained CRF emits major/minor only.
# 7th detection (if added later) should be a post-pass over these segments.

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
_CHORDS_VENV_PYTHON = _PROJECT_ROOT / "venv-chords" / "Scripts" / "python.exe"
_CHORD_DETECT_SCRIPT = _PROJECT_ROOT / "chord_detect.py"


def detect_chords(audio_path: str) -> list[dict]:
    """
    Detect chords by invoking chord_detect.py inside the Python 3.10 sidecar
    venv (venv-chords/), which has madmom installed.

    The sidecar runs madmom's CNNChordFeatureProcessor + CRFChordRecognitionProcessor
    on the stem and emits JSON segments {chord, start, end} on stdout.

    Returns list of dicts: {chord, start, end}. "N" = no chord (silence/break).
    """
    log.info("Stage 3: detecting chords with madmom (sidecar venv) from %s", audio_path)
    print("[3/4] Detecting chords (madmom CNN+CRF)...")

    if not _CHORDS_VENV_PYTHON.exists():
        raise FileNotFoundError(
            f"Sidecar venv not found at {_CHORDS_VENV_PYTHON}. "
            f"Run setup-chords-venv.bat once to create it."
        )
    if not _CHORD_DETECT_SCRIPT.exists():
        raise FileNotFoundError(
            f"Sidecar script not found at {_CHORD_DETECT_SCRIPT}."
        )

    proc = subprocess.run(
        [str(_CHORDS_VENV_PYTHON), str(_CHORD_DETECT_SCRIPT), str(audio_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    # Sidecar uses stderr for progress messages — relay them to our log
    if proc.stderr:
        for line in proc.stderr.splitlines():
            if line.strip():
                log.info("[chord_detect] %s", line)
                print(f"    {line}")

    if proc.returncode != 0:
        raise RuntimeError(
            f"chord_detect.py exited {proc.returncode}.\n"
            f"stderr:\n{proc.stderr}"
        )

    try:
        segments = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Could not parse chord_detect.py output as JSON: {e}\n"
            f"stdout:\n{proc.stdout[:500]}"
        )

    real = [c for c in segments if c["chord"] != "N"]
    log.info("Detected %d chord segments (%d total inc. silence)", len(real), len(segments))
    print(f"    Detected {len(real)} chord segments (excluding silences).")
    return segments


# ---------------------------------------------------------------------------
# Stage 4 — Alignment + ChordPro formatting
# ---------------------------------------------------------------------------

# Small lookahead to catch words that start just before a guitar strum.
# Onset timing is accurate but vocals often land a fraction of a second
# before the downstroke — 0.1s pulls the annotation forward without overcorrecting.
CHORD_LOOKAHEAD = 0.1


def _chord_at(chords: list[dict], time: float) -> str | None:
    """Return the chord active at `time`, with a short lookahead for upcoming changes."""
    current = None
    for c in chords:
        if c["start"] <= time < c["end"]:
            current = None if c["chord"] == "N" else c["chord"]
            break

    for c in chords:
        if time < c["start"] <= time + CHORD_LOOKAHEAD and c["chord"] not in ("N", current):
            return c["chord"]

    return current


def build_chordpro(
    lines: list[dict],
    chords: list[dict],
    title: str = "",
    artist: str = "",
) -> str:
    """
    Align chords to lyric lines and produce ChordPro text.
    Chords are annotated inline at the word where they change,
    not just at line starts.
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

    for line_idx, line in enumerate(lines):
        words = line.get("words", [])
        if not words:
            out.append(line["text"])
            continue

        parts = []
        for i, word in enumerate(words):
            text = word["word"].strip()
            chord = _chord_at(chords, word["start"])
            if chord and chord != prev_chord:
                prefix = f"[{chord}] " if not parts else f"[{chord}]"
                parts.append(f"{prefix}{text}")
                prev_chord = chord
            else:
                parts.append(text)

        # Append any chord changes that fall in the gap after this line's last word
        gap_start = line["end"]
        gap_end = lines[line_idx + 1]["start"] if line_idx + 1 < len(lines) else gap_start
        for c in chords:
            if gap_start <= c["start"] < gap_end and c["chord"] not in ("N", prev_chord):
                parts.append(f"[{c['chord']}]")
                prev_chord = c["chord"]

        out.append(" ".join(parts))

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

    vocals_path, other_path = separate_stems(audio_path, work_dir)
    words = transcribe(vocals_path)
    lines = group_into_lines(words)
    chords = detect_chords(other_path)   # use guitar/piano stem, not full mix
    chopro = build_chordpro(lines, chords, title=title, artist=artist)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(chopro)

    log.info("Saved ChordPro to %s", output_path)
    print(f"\nSaved: {output_path}")
    return chopro
