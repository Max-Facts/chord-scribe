chord-scribe
============
Converts an audio recording of a song into a ChordPro (.chopro) file
with lyrics and chords aligned — like a chord/lyric sheet.


REQUIREMENTS
------------
Python 3.10+

Install dependencies:
  pip install -r requirements.txt


FIRST RUN
---------
The first run will automatically download:
  - htdemucs model weights (Demucs, ~80 MB)
  - faster-whisper large-v2 model (~3 GB)
  - autochord model weights

This is expected. Subsequent runs use the cached models.


CLI USAGE
---------
  python main.py <audio_file> [--title "Song Title"] [--artist "Artist Name"]

Examples:
  python main.py song.mp3
  python main.py song.wav --title "Blackbird" --artist "The Beatles"

Output is saved as a .chopro file in the same directory as the input,
and a preview is printed to the console.


OUTPUT FORMAT
-------------
ChordPro (.chopro) — plain text with inline chord markers, e.g.:

  {title: Blackbird}
  {artist: The Beatles}

  [G]Blackbird singing in the [Am]dead of night
  [C]Take these broken wings and [G]learn to fly

Chord accuracy is ~70-85%. The output is a starting point for human
review, not a finished document.


TUNING
------
The line-grouping gap threshold (default 1.0 seconds) can be adjusted
in pipeline.py:

  LINE_GAP_THRESHOLD = 1.0
