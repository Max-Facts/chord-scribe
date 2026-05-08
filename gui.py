"""
chord-scribe GUI — CustomTkinter
Run the pipeline in a background thread so the UI stays responsive.
"""

import logging
import pathlib
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

import pipeline

# ── Appearance ───────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

log = logging.getLogger(__name__)

STAGES = [
    "Separating stems...",
    "Transcribing vocals...",
    "Detecting chords...",
    "Aligning and formatting...",
    "Done.",
]


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("chord-scribe")
        self.geometry("700x620")
        self.resizable(False, False)

        self._chopro_text = ""
        self._output_path = ""

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 16, "pady": 6}

        # Audio file
        ctk.CTkLabel(self, text="Audio file", anchor="w").pack(fill="x", **pad)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(0, 6))
        self.audio_entry = ctk.CTkEntry(row, placeholder_text="Select audio file...")
        self.audio_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row, text="Browse", width=80, command=self._pick_audio).pack(side="left")

        # Title / Artist
        meta = ctk.CTkFrame(self, fg_color="transparent")
        meta.pack(fill="x", padx=16, pady=(0, 6))
        meta.columnconfigure(0, weight=1)
        meta.columnconfigure(1, weight=1)

        ctk.CTkLabel(meta, text="Title", anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(meta, text="Artist", anchor="w").grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.title_entry = ctk.CTkEntry(meta, placeholder_text="Song title")
        self.title_entry.grid(row=1, column=0, sticky="ew")
        self.artist_entry = ctk.CTkEntry(meta, placeholder_text="Artist name")
        self.artist_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        # Output folder
        ctk.CTkLabel(self, text="Output folder", anchor="w").pack(fill="x", **pad)
        row2 = ctk.CTkFrame(self, fg_color="transparent")
        row2.pack(fill="x", padx=16, pady=(0, 6))
        self.output_entry = ctk.CTkEntry(row2, placeholder_text="Same folder as audio file")
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row2, text="Browse", width=80, command=self._pick_output).pack(side="left")

        # Generate button
        self.generate_btn = ctk.CTkButton(
            self, text="Generate", height=40, command=self._start_pipeline
        )
        self.generate_btn.pack(fill="x", padx=16, pady=(10, 6))

        # Progress bar + status label
        self.status_label = ctk.CTkLabel(self, text="", anchor="w", text_color="gray")
        self.status_label.pack(fill="x", padx=16)
        self.progress = ctk.CTkProgressBar(self)
        self.progress.pack(fill="x", padx=16, pady=(4, 10))
        self.progress.set(0)

        # Preview pane
        ctk.CTkLabel(self, text="ChordPro preview", anchor="w").pack(fill="x", padx=16)
        self.preview = ctk.CTkTextbox(self, height=260, font=("Courier New", 12))
        self.preview.pack(fill="both", expand=True, padx=16, pady=(0, 6))
        self.preview.configure(state="disabled")

        # Save button
        self.save_btn = ctk.CTkButton(
            self, text="Save .chopro", height=36, state="disabled", command=self._save
        )
        self.save_btn.pack(fill="x", padx=16, pady=(0, 14))

    # ── File pickers ──────────────────────────────────────────────────────────

    def _pick_audio(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aiff"), ("All files", "*.*")],
        )
        if path:
            self.audio_entry.delete(0, "end")
            self.audio_entry.insert(0, path)
            # Auto-fill output folder to match audio file location
            if not self.output_entry.get():
                self.output_entry.insert(0, str(pathlib.Path(path).parent))

    def _pick_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _start_pipeline(self):
        audio_path = self.audio_entry.get().strip()
        if not audio_path or not pathlib.Path(audio_path).exists():
            messagebox.showerror("Error", "Please select a valid audio file.")
            return

        self.generate_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.progress.set(0)
        self._set_status("Starting...")
        self._clear_preview()

        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self):
        audio_path = self.audio_entry.get().strip()
        title  = self.title_entry.get().strip()
        artist = self.artist_entry.get().strip()

        out_dir = self.output_entry.get().strip()
        if out_dir:
            stem = pathlib.Path(audio_path).stem
            output_path = str(pathlib.Path(out_dir) / f"{stem}.chopro")
        else:
            output_path = str(pathlib.Path(audio_path).with_suffix(".chopro"))

        self._output_path = output_path

        # Patch pipeline stage functions to report progress
        original_separate = pipeline.separate_stems
        original_transcribe = pipeline.transcribe
        original_chords = pipeline.detect_chords
        original_build = pipeline.build_chordpro

        def _sep(*a, **kw):
            self._set_progress(0.05, STAGES[0])
            result = original_separate(*a, **kw)
            self._set_progress(0.30)
            return result

        def _trans(*a, **kw):
            self._set_progress(0.35, STAGES[1])
            result = original_transcribe(*a, **kw)
            self._set_progress(0.60)
            return result

        def _chords(*a, **kw):
            self._set_progress(0.65, STAGES[2])
            result = original_chords(*a, **kw)
            self._set_progress(0.85)
            return result

        def _build(*a, **kw):
            self._set_progress(0.90, STAGES[3])
            result = original_build(*a, **kw)
            return result

        pipeline.separate_stems   = _sep
        pipeline.transcribe       = _trans
        pipeline.detect_chords    = _chords
        pipeline.build_chordpro   = _build

        try:
            chopro = pipeline.process(
                audio_path=audio_path,
                output_path=output_path,
                title=title,
                artist=artist,
            )
            self._chopro_text = chopro
            self.after(0, self._on_success)
        except Exception as e:
            log.exception("Pipeline error: %s", e)
            err_msg = str(e)  # bind before lambda; e is cleared on except exit
            self.after(0, lambda: self._on_error(err_msg))
        finally:
            pipeline.separate_stems  = original_separate
            pipeline.transcribe      = original_transcribe
            pipeline.detect_chords   = original_chords
            pipeline.build_chordpro  = original_build

    def _on_success(self):
        self._set_progress(1.0, STAGES[4])
        self._show_preview(self._chopro_text)
        self.generate_btn.configure(state="normal")
        self.save_btn.configure(state="normal")

    def _on_error(self, msg: str):
        self._set_status(f"Error: {msg}")
        self.progress.set(0)
        self.generate_btn.configure(state="normal")
        messagebox.showerror("Pipeline error", msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, text: str):
        self.after(0, lambda: self.status_label.configure(text=text))

    def _set_progress(self, value: float, status: str = ""):
        if status:
            self._set_status(status)
        self.after(0, lambda: self.progress.set(value))

    def _clear_preview(self):
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.configure(state="disabled")

    def _show_preview(self, text: str):
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def _save(self):
        if not self._chopro_text:
            return
        path = filedialog.asksaveasfilename(
            title="Save ChordPro file",
            defaultextension=".chopro",
            initialfile=pathlib.Path(self._output_path).name if self._output_path else "output.chopro",
            filetypes=[("ChordPro", "*.chopro"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._chopro_text)
            self._set_status(f"Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    log_path = pathlib.Path(__file__).parent / "chord-scribe.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger(__name__).info("GUI started, logging to %s", log_path)
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
