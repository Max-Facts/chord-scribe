# Changelog

All notable changes to chord-scribe will be documented here.

Format: [version] — date — description

---

## [Unreleased]

### Session 4 — 2026-05-06

#### Pipeline
- Replaced librosa onset+chroma+template chord detection with **madmom**
  (`CNNChordFeatureProcessor` + `CRFChordRecognitionProcessor`). The CRF
  handles segmentation and smoothing internally, so the onset detection,
  template matching, `CHORD_MIN_DURATION` filter, and `CHORD_LOOKAHEAD`
  hack are all gone.
- Vocabulary remains major/minor only — that's what madmom's pretrained
  CRF model emits. 7th detection (post-pass over madmom segments using
  chroma analysis) is parked for a future session.

#### Architecture: sidecar venv
- madmom doesn't build on Python 3.14 (the main venv). Chord detection
  now runs in a separate Python 3.8 venv at `venv-chords/`, isolated
  from the main pipeline.
- New `chord_detect.py` runs inside the sidecar venv: takes audio path
  on argv, prints JSON segments to stdout, progress to stderr.
- `pipeline.py:detect_chords()` shells out to it via `subprocess.run`.
- New `setup-chords-venv.bat` provisions the sidecar venv from
  `C:\Users\admin\AppData\Local\Programs\Python\Python38\python.exe`,
  pinning `numpy<1.24`, `scipy<1.13`, `cython<3.0`, `mido<1.3`,
  `madmom==0.16.1`. Run once after pulling.
- New `requirements-chords.txt` mirrors the sidecar deps for documentation.

#### Cleanup
- `tune.py` ITERATIONS reduced to line-grouping params only
  (`LINE_GAP_THRESHOLD`, `LINE_MAX_WORDS`); chord-detection tunables
  are now inside `chord_detect.py`.

### Session 3 — 2026-04-23

#### Pipeline
- Switched chord detection from beat-synchronous to onset-synchronous: boundaries now
  use `librosa.onset.onset_detect` on the isolated guitar/piano stem — actual note
  attacks rather than metrical beats
- Reverted chord templates to major/minor only — expanded types (7th, sus, dim, maj7)
  caused false positives due to chroma ambiguity
- CHORD_MIN_DURATION raised to 0.5s to suppress rapid-strum micro-segments
- CHORD_LOOKAHEAD reduced to 0.1s (was 0.25s); corrects vocal-before-strum timing
  offset; still slightly overcorrecting — further tuning needed
- LINE_MAX_WORDS raised from 8 to 14 to avoid mid-phrase line breaks
- Gap chord detection: chord changes occurring in silence after a line's last word
  are now appended as standalone [chord] annotations (e.g. `[F]fool [A#]`)
- ChordPro formatting: space after chord bracket when opening a line (`[F] Never`)
  to reflect chord sounding before first syllable; inline chords unchanged (`[Dm]me`)

#### GUI / launcher
- Added `run.bat` double-click launcher — activates venv and runs gui.py
- GUI now writes a persistent `chord-scribe.log` file in the project folder
  so crash tracebacks survive after the window closes

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
