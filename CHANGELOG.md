# Changelog

All notable changes to chord-scribe will be documented here.

Format: [version] — date — description

---

## [Unreleased]

### Session 5 — 2026-05-07

#### Pipeline — line break tuning
- `group_into_lines()` now takes optional `chords` parameter. When chord
  segments are passed in, a chord change at a word boundary becomes a
  soft phrase-break signal (gated by `LINE_MIN_WORDS_BEFORE_CHORD_BREAK`,
  default 8). Lyric phrases and chord progressions tend to align in pop
  songs, so this catches phrase ends Whisper's punctuation/capitalization
  signals miss.
- New `LINE_MIN_WORDS_FOR_SOFT_BREAK` constant (default 3) gates ALL soft
  break signals (gap, chord-change, punctuation/capitalization). A break
  only fires if the closing line would have at least this many words.
  Prevents 1–2 word orphan fragments ("pool", "cruel", "news") that sung
  vocals' breath-pause gaps were producing. The hard `LINE_MAX_WORDS`
  cap still wins regardless.
- `process()` now runs chord detection BEFORE line grouping, so chord
  segments are available for the chord-aware break logic.
- `tune.py` ITERATIONS now exercises chord_min and the gate together.

#### Validated on Golden Fool
- 44 generated lines vs 47 in manual reference. Section-boundary tag
  lines (`Yeah, golden fool` etc.) now correctly stand alone instead of
  jamming into the next verse.
- All chord roots (F, Dm, A#, Gm) match manual on every appearance.
- C and Am7 in manual aren't on the recording (Max's bass parts) — not
  a detection miss.

#### Sidecar Python downgrade
- Provisioned `venv-chords/` against Python 3.8 (originally aimed at
  3.10, but only 3.7/3.8 were installed locally). madmom 0.16.1 builds
  cleanly on 3.8 with the same pinned numpy/scipy/cython/mido.

#### Bug fixes
- gui.py: bind exception message to a local variable before passing to
  `self.after()` lambda; previously `e` was cleared at except-block exit
  and the lambda raised NameError when an actual pipeline error fired.
- chord_detect.py: added `from __future__ import annotations` for
  Python 3.8 compatibility with `list[dict]` style return annotations.

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
- Word-level