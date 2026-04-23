"""
chord-scribe pipeline
Stages: source separation → transcription → chord detection → align + format
"""

import logging
import pathlib
import numpy as np
import librosa
import soundfile as sf
import torch
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

# Tunable constants
LINE_GAP_THRESHOLD = 0.5  # seconds of silence that starts a new lyric line
LINE_MAX_WORDS = 8        # max words per line before forcing a break (fallback for sung lyrics)


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

    log.info("Stage 1: separating stems with Demucs htdemucs")
    print("[1/4] Separating stems (Demucs htdemucs)...")

    wav_np, sr = sf.read(str(audio_path), always_2d=True)  # (samples, channels)
    wav_np = wav_np.T.astype(np.float32)                   # (channels, samples)
    wav = torch.from_numpy(wav_np)

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    model = get_model("htdemucs")
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
# Stage 3 — Chord detection (librosa chroma + template matching)
# ---------------------------------------------------------------------------

# Chord templates: 12 chroma bins starting from C
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _build_templates() -> dict[str, np.ndarray]:
    """Return major and minor chord templates (unit vectors over 12 chroma bins)."""
    major_intervals = [0, 4, 7]   # root, major 3rd, perfect 5th
    minor_intervals = [0, 3, 7]   # root, minor 3rd, perfect 5th

    templates = {}
    for i, note in enumerate(_NOTES):
        maj = np.zeros(12)
        for interval in major_intervals:
            maj[(i + interval) % 12] = 1.0
        templates[note] = maj / np.linalg.norm(maj)

        min_ = np.zeros(12)
        for interval in minor_intervals:
            min_[(i + interval) % 12] = 1.0
        templates[f"{note}m"] = min_ / np.linalg.norm(min_)

    return templates

_TEMPLATES = _build_templates()

# How many seconds per chord analysis frame
CHORD_HOP_SECONDS = 0.25
# Minimum chord duration to emit (merges tiny fragments)
CHORD_MIN_DURATION = 0.5
# Energy threshold below which we emit "N" (silence/no chord)
CHORD_ENERGY_THRESHOLD = 0.005


def _best_chord(chroma_frame: np.ndarray) -> str:
    """Return the chord label with the highest cosine similarity to chroma_frame."""
    chroma_frame = chroma_frame / (np.linalg.norm(chroma_frame) + 1e-8)
    best_label = "N"
    best_score = -1.0
    for label, template in _TEMPLATES.items():
        score = float(np.dot(chroma_frame, template))
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def detect_chords(audio_path: str) -> list[dict]:
    """
    Detect chords from the given audio file using librosa chroma features.
    For best results, pass the 'other' stem (guitar/piano) rather than the
    original mix — bass and drums interfere with chroma analysis.
    Returns list of dicts: {chord, start, end}
    """
    log.info("Stage 3: detecting chords from %s", audio_path)
    print("[3/4] Detecting chords (librosa chroma template matching)...")

    y, sr = librosa.load(audio_path, mono=True)
    hop_length = int(sr * CHORD_HOP_SECONDS)

    # CQT-based chromagram is more accurate for chord detection than STFT
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    n_frames = chroma.shape[1]
    frame_chords = []
    for i in range(n_frames):
        t = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
        energy = float(rms[min(i, len(rms) - 1)])
        if energy < CHORD_ENERGY_THRESHOLD:
            label = "N"
        else:
            label = _best_chord(chroma[:, i])
        frame_chords.append((t, label))

    # Merge consecutive identical chords into segments
    segments = []
    if frame_chords:
        seg_start, seg_label = frame_chords[0]
        for t, label in frame_chords[1:]:
            if label != seg_label:
                segments.append({"chord": seg_label, "start": seg_start, "end": t})
                seg_start, seg_label = t, label
        total_dur = librosa.get_duration(y=y, sr=sr)
        segments.append({"chord": seg_label, "start": seg_start, "end": total_dur})

    # Merge very short segments into the previous one
    merged = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if merged and dur < CHORD_MIN_DURATION:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    real = [c for c in merged if c["chord"] != "N"]
    log.info("Detected %d chord segments (%d total inc. silence)", len(real), len(merged))
    print(f"    Detected {len(real)} chord segments (excluding silences).")
    return merged


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

    for line in lines:
        words = line.get("words", [])
        if not words:
            out.append(line["text"])
            continue

        parts = []
        for word in words:
            text = word["word"].strip()
            chord = _chord_at(chords, word["start"])
            if chord and chord != prev_chord:
                parts.append(f"[{chord}]{text}")
                prev_chord = chord
            else:
                parts.append(text)

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
