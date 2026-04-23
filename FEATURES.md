# Planned Features

Features to implement when building the GUI (and pipeline support where needed).

---

## Capo input
- Number field in UI (0–12)
- Transposes all detected chords down by N semitones before writing ChordPro
- Adds `{capo: N}` directive to the ChordPro header
- Pipeline: add optional `capo` parameter to `build_chordpro()`

## Existing lyrics (.txt file)
- Optional file picker in UI for a pre-written lyrics file
- Whisper still runs for word-level timestamps — the .txt replaces the
  transcribed *text* but timing still comes from Whisper's alignment
- Approach: run Whisper with forced alignment against the provided lyrics,
  or align provided lines to Whisper word timestamps by sequence matching
- Useful for: correcting Whisper mishears, songs with tricky vocals
- Pipeline: `process()` should accept an optional `lyrics_path` parameter

## Artist / song autofill from audio metadata
- On audio file selection, read ID3/FLAC/etc. tags using `mutagen`
- Pre-populate Title and Artist fields from embedded metadata
- Fields remain editable by the user
- Graceful fallback to blank if no tags found
- Add `mutagen` to requirements.txt when implementing
