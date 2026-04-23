"""
chord-scribe CLI entry point

Usage:
  python main.py <audio_file> [--title "Song Title"] [--artist "Artist Name"]
"""

import argparse
import logging
import pathlib
import sys
from pipeline import process


def setup_logging(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


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

    output_path = args.output or str(audio_path.with_suffix(".chopro"))
    log_path = str(audio_path.with_suffix(".log"))
    setup_logging(log_path)

    log = logging.getLogger(__name__)
    log.info("chord-scribe started: %s", audio_path)

    try:
        chopro = process(
            audio_path=str(audio_path),
            output_path=output_path,
            title=args.title,
            artist=args.artist,
        )
        print("\n--- ChordPro Preview ---")
        print(chopro)
        print("------------------------")
        log.info("Done.")
    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        print(f"\nError: {e}")
        print(f"See {log_path} for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
