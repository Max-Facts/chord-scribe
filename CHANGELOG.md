# Changelog

All notable changes to chord-scribe will be documented here.

Format: [version] — date — description

---

## [Unreleased]

### Session 2 — 2026-04-23

#### Pipeline
- Beat-synchronous chord detection: chords analyzed per musical beat via librosa
  beat tracker; boundaries now align with actual musical events
- Chord lookahead (0.25s): corrects systematic 1-word-late bias caused by words
  starting just before the beat where the chord changes
- Word-level chord annotation: chords appear inline at the exact word where they
  change, not just at line starts
- Replaced autochord (vamp/Windows broken) with pure librosa chroma template matching
- Replaced torchaudio I/O (requires torchcodec) with soundfile for all audio I/O
- Use Demucs 'other' stem (guitar/piano) for chord detection — removes bass/drum
  interference from chroma analysis
- Added CUDA detection to Demucs stage (Whisper already had it); GPU gives ~10x
  speedup on RTX 3060 (full run ~30s vs ~5min on CPU)
- Installed PyTorch CUDA build (cu126) to activate RTX 3060
- LINE_MAX_WORDS=8 fallback for line grouping (sung lyrics have few natural pauses)
- Punctuation-aware line breaks; capitalization used as soft phrase-boundary signal
- Tuning script (tune.py) for fast parameter sweeps with cached ML data

#### Logging
- pipeline.py: logging throughout all 4 stages
- main.py: errors caught and written to a .log file alongside .chopro output

#### GUI
- gui.py: CustomTkinter interface with file picker, title/artist fields,
  output folder picker, progress bar, ChordPro preview pane, and Save button
- Pipeline runs in background thread — UI stays responsive during processing

### Session 1 — 2026-04-23

- Project initialized
- Pipeline scaffolded: Demucs → faster-whisper → chord detection → ChordPro
- requirements.txt, README.txt, CHANGELOG.md, .gitignore
- GitHub repo created and connected (Max-Facts/chord-scribe)
