"""
Text to Sign Language (T2SL) - Interactive Visual Player Application
Translates written text or microphone speech into sequential Sign Language gestures
with real-time animated playback and visual landmark diagrams.
"""

import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

from src.text_to_sign import TextToSignEngine
from src.tts_engine import TTSEngine


class TextToSignApp(tk.Tk):
    """Desktop player for translating Text or Voice into Sign Language."""

    def __init__(self):
        super().__init__()

        self.title("Text & Voice to Sign Language - Assistive Visual Translator")
        self.geometry("1080x780")
        self.minsize(950, 700)
        self.configure(bg="#1a1c23")

        self.engine = TextToSignEngine()
        self.tts = TTSEngine(rate=160, volume=1.0)

        # State management
        self.tokens = []
        self.current_index = 0
        self.is_playing = False
        self.playback_speed = 0.9  # seconds per sign
        self.speak_while_signing = tk.BooleanVar(value=True)

        self._build_header()
        self._build_input_panel()
        self._build_player_panel()
        self._build_timeline_panel()

        # Set default welcome translation
        self.translate_and_load("Hello thank you")

    def _build_header(self):
        header = tk.Frame(self, bg="#262a36", height=60)
        header.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(
            header,
            text="TEXT & VOICE TO SIGN LANGUAGE TRANSLATOR",
            font=("Segoe UI", 15, "bold"),
            fg="#00e5ff",
            bg="#262a36"
        )
        title.pack(side=tk.LEFT, padx=20, pady=12)

        subtitle = tk.Label(
            header,
            text="Bridging Hearing & Deaf Communication with Interactive Visual Signs",
            font=("Segoe UI", 10, "italic"),
            fg="#9aa0b4",
            bg="#262a36"
        )
        subtitle.pack(side=tk.LEFT, padx=5, pady=16)

    def _build_input_panel(self):
        input_frame = tk.LabelFrame(
            self,
            text=" 1. Enter Text or Speak via Microphone ",
            font=("Segoe UI", 11, "bold"),
            fg="#9aa0b4",
            bg="#20232c",
            bd=1,
            relief=tk.SOLID
        )
        input_frame.pack(fill=tk.X, padx=20, pady=(15, 10))

        entry_row = tk.Frame(input_frame, bg="#20232c")
        entry_row.pack(fill=tk.X, padx=15, pady=12)

        self.text_entry = tk.Entry(
            entry_row,
            font=("Segoe UI", 13),
            bg="#13151b",
            fg="#ffffff",
            insertbackground="white",
            bd=1,
            relief=tk.SOLID
        )
        self.text_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.text_entry.insert(0, "Hello, can you please help me?")
        self.text_entry.bind("<Return>", lambda event: self.on_translate_click())

        trans_btn = tk.Button(
            entry_row,
            text="Translate to Signs",
            command=self.on_translate_click,
            bg="#00b4d8",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=16,
            pady=6,
            cursor="hand2"
        )
        trans_btn.pack(side=tk.LEFT, padx=5)

        self.mic_btn = tk.Button(
            entry_row,
            text="🎙️ Voice Input",
            command=self.start_mic_transcription,
            bg="#7209b7",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2"
        )
        self.mic_btn.pack(side=tk.LEFT, padx=5)

        # Quick phrases bar
        phrases_frame = tk.Frame(input_frame, bg="#20232c")
        phrases_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        tk.Label(phrases_frame, text="Quick Signs:", font=("Segoe UI", 9, "bold"), fg="#9aa0b4", bg="#20232c").pack(side=tk.LEFT, padx=(0, 8))
        common_phrases = ["Hello", "Thank you", "Help", "Water", "Yes", "No", "I love you", "Please"]
        for p in common_phrases:
            b = tk.Button(
                phrases_frame,
                text=p,
                command=lambda text=p: self.load_quick_phrase(text),
                bg="#2e3442",
                fg="#e0e6ed",
                font=("Segoe UI", 8),
                relief=tk.FLAT,
                padx=8,
                pady=2,
                cursor="hand2"
            )
            b.pack(side=tk.LEFT, padx=3)

    def _build_player_panel(self):
        player_frame = tk.Frame(self, bg="#1a1c23")
        player_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        # Center sign display card
        self.display_canvas = tk.Label(player_frame, bg="#13151b", bd=2, relief=tk.SOLID)
        self.display_canvas.pack(side=tk.TOP, pady=5)

        # Playback Controls Row
        ctrl_row = tk.Frame(player_frame, bg="#1a1c23")
        ctrl_row.pack(fill=tk.X, pady=10)

        self.prev_btn = tk.Button(
            ctrl_row, text="⏮ Prev", command=self.prev_sign,
            bg="#3a3f50", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=12, pady=5, cursor="hand2"
        )
        self.prev_btn.pack(side=tk.LEFT, padx=6)

        self.play_btn = tk.Button(
            ctrl_row, text="▶ Play Sequence", command=self.toggle_play,
            bg="#06d6a0", fg="#1a1c23", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=18, pady=5, cursor="hand2"
        )
        self.play_btn.pack(side=tk.LEFT, padx=6)

        self.next_btn = tk.Button(
            ctrl_row, text="Next ⏭", command=self.next_sign,
            bg="#3a3f50", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=12, pady=5, cursor="hand2"
        )
        self.next_btn.pack(side=tk.LEFT, padx=6)

        # Speed slider
        tk.Label(ctrl_row, text="Sign Duration (s):", font=("Segoe UI", 9), fg="#9aa0b4", bg="#1a1c23").pack(side=tk.LEFT, padx=(20, 5))
        self.speed_slider = tk.Scale(
            ctrl_row, from_=0.4, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
            bg="#1a1c23", fg="white", highlightthickness=0, command=self.on_speed_change
        )
        self.speed_slider.set(0.9)
        self.speed_slider.pack(side=tk.LEFT, padx=5)

        # Audio narration checkbox
        audio_chk = tk.Checkbutton(
            ctrl_row,
            text="Narrate with Audio Speech",
            variable=self.speak_while_signing,
            bg="#1a1c23",
            fg="#e0e6ed",
            selectcolor="#20232c",
            activebackground="#1a1c23",
            font=("Segoe UI", 9)
        )
        audio_chk.pack(side=tk.RIGHT, padx=10)

    def _build_timeline_panel(self):
        timeline_frame = tk.LabelFrame(
            self,
            text=" 2. Translation Sequence Timeline ",
            font=("Segoe UI", 10, "bold"),
            fg="#9aa0b4",
            bg="#20232c",
            bd=1,
            relief=tk.SOLID
        )
        timeline_frame.pack(fill=tk.X, padx=20, pady=(5, 15))

        self.timeline_label = tk.Label(
            timeline_frame,
            text="(No sentence loaded)",
            font=("Segoe UI", 11),
            fg="#00e5ff",
            bg="#20232c",
            wraplength=980,
            justify=tk.LEFT
        )
        self.timeline_label.pack(fill=tk.X, padx=15, pady=8)

    def load_quick_phrase(self, text: str):
        self.text_entry.delete(0, tk.END)
        self.text_entry.insert(0, text)
        self.translate_and_load(text)

    def on_translate_click(self):
        text = self.text_entry.get().strip()
        if text:
            self.translate_and_load(text)
        else:
            messagebox.showwarning("Empty Input", "Please enter some text or click Voice Input.")

    def translate_and_load(self, text: str):
        self.is_playing = False
        self.play_btn.config(text="▶ Play Sequence", bg="#06d6a0")
        self.tokens = self.engine.translate_text(text)
        self.current_index = 0

        if not self.tokens:
            self.timeline_label.config(text="No recognizable sign tokens found.")
            return

        # Update timeline display
        token_strs = [f"[{t['token']}]" for t in self.tokens]
        self.timeline_label.config(text="  ➔  ".join(token_strs))

        # Show first sign
        self.display_current_sign()

        # Speak initial audio if enabled
        if self.speak_while_signing.get():
            self.tts.speak(text)

    def display_current_sign(self):
        if not self.tokens or not (0 <= self.current_index < len(self.tokens)):
            return

        current_token = self.tokens[self.current_index]
        token_name = current_token['token']

        # Render image
        bgr_img = self.engine.render_sign_image(token_name, width=440, height=380)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        img_tk = ImageTk.PhotoImage(image=pil_img)

        self.display_canvas.img_tk = img_tk
        self.display_canvas.config(image=img_tk)

        # Highlight in timeline
        display_parts = []
        for idx, t in enumerate(self.tokens):
            if idx == self.current_index:
                display_parts.append(f"⭐[{t['token']}]⭐")
            else:
                display_parts.append(f"[{t['token']}]")
        self.timeline_label.config(text="  ➔  ".join(display_parts))

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ Play Sequence", bg="#06d6a0")
        else:
            self.is_playing = True
            self.play_btn.config(text="⏸ Pause", bg="#f72585")
            threading.Thread(target=self._run_playback_loop, daemon=True).start()

    def _run_playback_loop(self):
        while self.is_playing and self.tokens:
            self.display_current_sign()
            time.sleep(self.playback_speed)

            if not self.is_playing:
                break

            self.current_index += 1
            if self.current_index >= len(self.tokens):
                self.current_index = 0
                self.is_playing = False
                self.after(0, lambda: self.play_btn.config(text="▶ Replay Sequence", bg="#06d6a0"))
                break

    def prev_sign(self):
        self.is_playing = False
        self.play_btn.config(text="▶ Play Sequence", bg="#06d6a0")
        if self.tokens:
            self.current_index = (self.current_index - 1) % len(self.tokens)
            self.display_current_sign()

    def next_sign(self):
        self.is_playing = False
        self.play_btn.config(text="▶ Play Sequence", bg="#06d6a0")
        if self.tokens:
            self.current_index = (self.current_index + 1) % len(self.tokens)
            self.display_current_sign()

    def on_speed_change(self, val):
        self.playback_speed = float(val)

    def start_mic_transcription(self):
        self.mic_btn.config(text="🎙️ Listening...", bg="#e63946")

        def worker():
            transcribed = self.engine.speech_to_text(timeout=5)
            self.after(0, lambda: self._handle_mic_result(transcribed))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_mic_result(self, result: str):
        self.mic_btn.config(text="🎙️ Voice Input", bg="#7209b7")
        if result.startswith("[Error]") or result.startswith("[Warning]"):
            messagebox.showwarning("Voice Input", result)
        else:
            self.text_entry.delete(0, tk.END)
            self.text_entry.insert(0, result)
            self.translate_and_load(result)


def main():
    app = TextToSignApp()
    app.mainloop()


if __name__ == "__main__":
    main()
