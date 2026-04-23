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

## Batch processing
- Allow selecting multiple audio files (or a folder) at once
- Queue displayed in the UI showing each file and its status (waiting, processing, done, error)
- Processes one file at a time sequentially (ML models are memory-heavy)
- Title/Artist fields become optional per-file overrides; autofill from metadata where available
- Output .chopro files saved alongside each source file (or to a shared output folder)
- CLI support: `python main.py folder/ --output-dir ./output/`

## Cancel button
- Stops the pipeline mid-run and re-enables the Generate button
- Demucs and Whisper don't support clean interruption natively; use a
  threading.Event flag checked between stages — can't cancel mid-stage
  but will stop before the next one starts
- Clean up any partial work files on cancel

## Open in ChordPro
- Button that opens the saved .chopro file directly in the system's
  registered ChordPro application (or default .chopro handler)
- Use `os.startfile()` on Windows, `subprocess.run(['open', ...])` on Mac
- Only enabled after a successful generation + save

## Real-time progress bar (Demucs tqdm relay)
- Currently the progress bar jumps at stage boundaries; Demucs takes the longest
- Intercept Demucs tqdm output and relay percentage to the GUI progress bar in real time
- Makes the separation stage feel active rather than frozen

## Packaging (PyInstaller)
- Bundle into a standalone .exe using PyInstaller --onedir mode
- Model weights (Demucs, Whisper) download at runtime — not bundled
- Ensure cache directory is persistent across runs and not inside any temp/extracted folder
- Requires CUDA torch — document GPU vs CPU build difference for packaged installs
- Test on a clean machine if possible

## Artist / song autofill from audio metadata
- On audio file selection, read ID3/FLAC/etc. tags using `mutagen`
- Pre-populate Title and Artist fields from embedded metadata
- Fields remain editable by the user
- Graceful fallback to blank if no tags found
- Add `mutagen` to requirements.txt when implementing
