"""
chord-scribe CLI entry point

Usage:
  python main.py <audio_file> [--title "Song Title"] [--artist "Artist Name"]
"""

import argparse
import pathlib
import sys
from pipeline import process


def main():
    parser = argparse.ArgumentParser(
        description="Convert an audio file to a ChordPro (.chopro) file."
    )
    parser.add_argument("audio_file", help="Path to the input audio file")
    parser.add_argument("--title", default="", help="Song title")
    parser.add_argument("--artist", default="", help="Artist name")
    parser.add_argument(
        "--output",
        default="",
        help="Output .chopro file path (default: same dir as input)",
    )
    args = parser.parse_args()

    audio_path = pathlib.Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        output_path = str(audio_path.with_suffix(".chopro"))

    chopro = process(
        audio_path=str(audio_path),
        output_path=output_path,
        title=args.title,
        artist=args.artist,
    )

    print("\n--- ChordPro Preview ---")
    print(chopro)
    print("------------------------")


if __name__ == "__main__":
    main()
