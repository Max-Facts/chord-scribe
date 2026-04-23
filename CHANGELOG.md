# Changelog

All notable changes to chord-scribe will be documented here.

Format: [version] — date — description

---

## [Unreleased]

### GUI
- Added gui.py: CustomTkinter interface with file picker, title/artist fields,
  output folder picker, progress bar, ChordPro preview pane, and Save button
- Pipeline runs in a background thread — UI stays responsive during processing
- Progress bar updates at each stage with status label

- Project initialized
- Pipeline complete: Demucs → faster-whisper → librosa chord detection → ChordPro output
- Replaced autochord (vamp dependency, broken on Windows) with pure librosa chroma template matching
- Replaced torchaudio I/O (requires torchcodec/shared FFmpeg) with soundfile
- Added LINE_MAX_WORDS fallback for line grouping (sung lyrics have few natural pauses)
- Tuned defaults: gap=0.5s, max_words=8, chord_hop=0.25s, min_dur=0.5s, energy=0.005
- Added tune.py for fast parameter iteration with cached transcription/chord data
