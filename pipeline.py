"""
chord-scribe pipeline
Stages: source separation → transcription → chord detection → align + format
"""

import os
import pathlib
import numpy as np
import librosa
import soundfile as sf
import torch
from faster_whisper import WhisperModel

# Tunable constants
LINE_GAP_THRESHOLD = 1.0  # seconds of silence that starts a new lyric line


# ---------------------------------------------------------------------------
# Stage 1 — Source separation (Demucs)
# ---------------------------------------------------------------------------

def separate_vocals(audio_path: str, output_dir: str) -> str:
    """
    Run Demucs htdemucs via Python API on audio_path, return path to the vocals stem.
    Uses soundfile for I/O to avoid torchaudio / torchcodec dependency issues.
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    audio_path = pathlib.Path(audio_path).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vocals_path = output_dir / f"{audio_path.stem}_vocals.wav"

    print("[1/4] Separating vocals (Demucs htdemucs)...")

    # Load audio with soundfile — no torchaudio/torchcodec needed
    wav_np, sr = sf.read(str(audio_path), always_2d=True)  # (samples, channels)
    wav_np = wav_np.T.astype(np.float32)                   # (channels, samples)
    wav = torch.from_numpy(wav_np)

    # Demucs htdemucs expects stereo (2 channels)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    model = get_model("htdemucs")
    model.eval()

    # Normalize
    ref = wav.mean(0)
    wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)

    with torch.no_grad():
        sources = apply_model(model, wav_norm[None], progress=True)[0]

    vocals_idx = model.sources.index("vocals")
    vocals_tensor = sources[vocals_idx]  # (channels, samples)

    # Denormalize
    vocals_tensor = vocals_tensor * (ref.std() + 1e-8) + ref.mean()

    # Save with soundfile
    vocals_np = vocals_tensor.cpu().numpy().T  # (samples, channels)
    sf.write(str(vocals_path), vocals_np, sr)

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
CHORD_HOP_SECONDS = 0.5
# Minimum chord duration to emit (merges tiny fragments)
CHORD_MIN_DURATION = 1.0
# Energy threshold below which we emit "N" (silence/no chord)
CHORD_ENERGY_THRESHOLD = 0.01


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
    Detect chords from the original audio file using librosa chroma features.
    Returns list of dicts: {chord, start, end}
    """
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
