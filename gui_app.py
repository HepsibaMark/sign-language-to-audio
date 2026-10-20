"""
Sign Language Assistive Suite - Desktop Bi-directional GUI Application
Integrates both directions into a unified communication hub:
  1. Sign Language to Audio (Camera ➔ Hand Tracking ➔ Speech Synthesis)
  2. Text & Voice to Sign Language (Text / Microphone ➔ Visual Sign Animations)
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.hand_detector import HandDetector
from src.gesture_classifier import GestureClassifier
from src.sentence_builder import SentenceBuilder
from src.tts_engine import TTSEngine
from src.text_to_sign import TextToSignEngine


class BiDirectionalSignApp(tk.Tk):
    """Unified desktop application for bi-directional deaf and hearing communication."""

    def __init__(self):
        super().__init__()

        self.title("Sign Language & Speech Assistive Suite - Bi-Directional Hub")
        self.geometry("1180x820")
        self.minsize(1050, 720)
        self.configure(bg="#1a1c23")

        # Core logic components
        self.detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.5)
        self.classifier = GestureClassifier()
        self.sentence_builder = SentenceBuilder(hold_time_seconds=0.8, cooldown_seconds=0.6)
        self.tts = TTSEngine(rate=160, volume=1.0)
        self.t2s_engine = TextToSignEngine()

        # Tab 1 state
        self.cap = None
        self.is_camera_running = False
        self.auto_speak_var = tk.BooleanVar(value=False)

        # Tab 2 state
        self.t2s_tokens = []
        self.t2s_current_index = 0
        self.t2s_is_playing = False
        self.t2s_speed = 0.9
        self.t2s_speak_audio = tk.BooleanVar(value=True)

        # Style configuration
        self._configure_styles()
        self._build_header()
        self._build_tabbed_interface()
        self._build_status_bar()

        # Auto-start camera for Tab 1
        self.start_camera()

        # Load initial sign for Tab 2
        self.t2s_translate_and_load("Hello thank you")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("TNotebook", background="#1a1c23", borderwidth=0)
        style.configure("TNotebook.Tab", background="#262a36", foreground="#9aa0b4",
                        padding=[18, 8], font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", "#00b4d8")],
                  foreground=[("selected", "#ffffff")])

    def _build_header(self):
        header = tk.Frame(self, bg="#20232c", height=65)
        header.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(
            header,
            text="BI-DIRECTIONAL SIGN LANGUAGE ASSISTIVE SUITE",
            font=("Segoe UI", 16, "bold"),
            fg="#00e5ff",
            bg="#20232c"
        )
        title.pack(side=tk.LEFT, padx=20, pady=12)

        subtitle = tk.Label(
            header,
            text="Two-Way Inclusive Communication: Sign ➔ Speech  |  Text & Voice ➔ Sign",
            font=("Segoe UI", 10, "italic"),
            fg="#8d99ae",
            bg="#20232c"
        )
        subtitle.pack(side=tk.LEFT, padx=5, pady=16)

    def _build_tabbed_interface(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))

        # Tab 1: Sign Language to Audio
        self.tab1 = tk.Frame(self.notebook, bg="#1a1c23")
        self.notebook.add(self.tab1, text="  🤟 1. Sign Language to Audio (Camera ➔ Speech)  ")
        self._build_tab1_content()

        # Tab 2: Text & Voice to Sign Language
        self.tab2 = tk.Frame(self.notebook, bg="#1a1c23")
        self.notebook.add(self.tab2, text="  ✍️ 2. Text & Voice to Sign Language (Text/Mic ➔ Sign)  ")
        self._build_tab2_content()

    # =========================================================================
    # TAB 1: SIGN LANGUAGE TO AUDIO
    # =========================================================================
    def _build_tab1_content(self):
        main_frame = tk.Frame(self.tab1, bg="#1a1c23")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Left Column: Video Feed
        left_col = tk.Frame(main_frame, bg="#1a1c23")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        video_card = tk.LabelFrame(
            left_col, text=" Live Camera Stream ", font=("Segoe UI", 10, "bold"),
            fg="#8d99ae", bg="#20232c", bd=1, relief=tk.SOLID
        )
        video_card.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(video_card, bg="#0d0e12")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        vid_btn_bar = tk.Frame(video_card, bg="#20232c")
        vid_btn_bar.pack(fill=tk.X, pady=(0, 8), padx=8)

        self.toggle_cam_btn = tk.Button(
            vid_btn_bar, text="Stop Camera", command=self.toggle_camera,
            bg="#d90429", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=10, pady=4, cursor="hand2"
        )
        self.toggle_cam_btn.pack(side=tk.LEFT, padx=5)

        auto_speak_chk = tk.Checkbutton(
            vid_btn_bar, text="Auto-speak confirmed signs", variable=self.auto_speak_var,
            bg="#20232c", fg="#edf2f4", selectcolor="#1a1c23", activebackground="#20232c", font=("Segoe UI", 9)
        )
        auto_speak_chk.pack(side=tk.LEFT, padx=15)

        # Right Column: Detection, Sentence, Settings
        right_col = tk.Frame(main_frame, bg="#1a1c23", width=420)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_col.pack_propagate(False)

        # Recognized Gesture Card
        det_card = tk.LabelFrame(
            right_col, text=" Recognized Gesture ", font=("Segoe UI", 10, "bold"),
            fg="#8d99ae", bg="#20232c", bd=1, relief=tk.SOLID
        )
        det_card.pack(fill=tk.X, pady=(0, 8))

        self.sign_display = tk.Label(
            det_card, text="NO HAND", font=("Segoe UI", 24, "bold"), fg="#4cc9f0", bg="#20232c"
        )
        self.sign_display.pack(pady=(6, 2))

        self.conf_display = tk.Label(
            det_card, text="Confidence: 0% | Model: None", font=("Segoe UI", 9), fg="#a0aab2", bg="#20232c"
        )
        self.conf_display.pack(pady=(0, 4))

        self.progress_bar = ttk.Progressbar(det_card, orient=tk.HORIZONTAL, length=380, mode='determinate')
        self.progress_bar.pack(pady=(2, 8), padx=15)

        # Sentence Builder Card
        sent_card = tk.LabelFrame(
            right_col, text=" Constructed Sentence ", font=("Segoe UI", 10, "bold"),
            fg="#8d99ae", bg="#20232c", bd=1, relief=tk.SOLID
        )
        sent_card.pack(fill=tk.X, pady=(0, 8))

        self.sentence_entry = tk.Entry(
            sent_card, font=("Segoe UI", 12), bg="#13151b", fg="#edf2f4", insertbackground="white", bd=1, relief=tk.SOLID
        )
        self.sentence_entry.pack(fill=tk.X, padx=12, pady=(8, 6))

        btn_row = tk.Frame(sent_card, bg="#20232c")
        btn_row.pack(fill=tk.X, padx=12, pady=4)

        speak_btn = tk.Button(
            btn_row, text="🔊 Speak", command=self.speak_current_sentence,
            bg="#4361ee", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=10, pady=5, cursor="hand2"
        )
        speak_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        tk.Button(btn_row, text="Space", command=self.add_space, bg="#3a0ca3", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=8, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=3)
        tk.Button(btn_row, text="Delete", command=self.delete_last, bg="#480ca8", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=8, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=3)
        tk.Button(btn_row, text="Clear", command=self.clear_sentence, bg="#560bad", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=8, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=(3, 0))

        # Audio Settings Card
        settings_card = tk.LabelFrame(
            right_col, text=" Voice Settings & Log ", font=("Segoe UI", 10, "bold"),
            fg="#8d99ae", bg="#20232c", bd=1, relief=tk.SOLID
        )
        settings_card.pack(fill=tk.BOTH, expand=True)

        voice_row = tk.Frame(settings_card, bg="#20232c")
        voice_row.pack(fill=tk.X, padx=12, pady=(6, 2))
        tk.Label(voice_row, text="Voice:", font=("Segoe UI", 8), fg="#edf2f4", bg="#20232c").pack(side=tk.LEFT)
        voice_names = [v['name'] for v in self.tts.available_voices] or ["Default"]
        self.voice_var = tk.StringVar(value=voice_names[0])
        voice_dropdown = ttk.Combobox(voice_row, textvariable=self.voice_var, values=voice_names, state="readonly", width=32)
        voice_dropdown.pack(side=tk.LEFT, padx=8)
        voice_dropdown.bind("<<ComboboxSelected>>", self.on_voice_changed)

        self.history_box = scrolledtext.ScrolledText(settings_card, height=5, bg="#13151b", fg="#4cc9f0", font=("Consolas", 9), bd=1, relief=tk.SOLID)
        self.history_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 8))

    # =========================================================================
    # TAB 2: TEXT & VOICE TO SIGN LANGUAGE
    # =========================================================================
    def _build_tab2_content(self):
        main_frame = tk.Frame(self.tab2, bg="#1a1c23")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Top: Input and Speech Recognition
        input_box = tk.LabelFrame(
            main_frame, text=" 1. Enter Text or Use Microphone Speech ",
            font=("Segoe UI", 10, "bold"), fg="#8d99ae", bg="#20232c", bd=1, relief=tk.SOLID
        )
        input_box.pack(fill=tk.X, pady=(0, 10))

        row1 = tk.Frame(input_box, bg="#20232c")
        row1.pack(fill=tk.X, padx=15, pady=10)

        self.t2s_entry = tk.Entry(row1, font=("Segoe UI", 12), bg="#13151b", fg="#ffffff", insertbackground="white", bd=1, relief=tk.SOLID)
        self.t2s_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.t2s_entry.insert(0, "Hello, thank you for your help!")
        self.t2s_entry.bind("<Return>", lambda event: self.t2s_translate_click())

        tk.Button(row1, text="Translate", command=self.t2s_translate_click, bg="#00b4d8", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=14, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=4)

        self.t2s_mic_btn = tk.Button(row1, text="🎙️ Speak Mic", command=self.t2s_start_mic, bg="#7209b7", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=12, pady=5, cursor="hand2")
        self.t2s_mic_btn.pack(side=tk.LEFT, padx=4)

        # Quick phrases
        q_frame = tk.Frame(input_box, bg="#20232c")
        q_frame.pack(fill=tk.X, padx=15, pady=(0, 8))
        tk.Label(q_frame, text="Quick Signs:", font=("Segoe UI", 8, "bold"), fg="#8d99ae", bg="#20232c").pack(side=tk.LEFT, padx=(0, 8))
        for p in ["Hello", "Thank you", "Help", "Water", "Yes", "No", "I love you", "Please"]:
            tk.Button(q_frame, text=p, command=lambda text=p: self.t2s_load_phrase(text), bg="#2e3442", fg="#e0e6ed", font=("Segoe UI", 8), relief=tk.FLAT, padx=6, pady=2, cursor="hand2").pack(side=tk.LEFT, padx=2)

        # Center: Sign Visual Display
        center_frame = tk.Frame(main_frame, bg="#1a1c23")
        center_frame.pack(fill=tk.BOTH, expand=True)

        self.t2s_canvas = tk.Label(center_frame, bg="#13151b", bd=2, relief=tk.SOLID)
        self.t2s_canvas.pack(side=tk.TOP, pady=6)

        # Controls Row
        ctrl_bar = tk.Frame(center_frame, bg="#1a1c23")
        ctrl_bar.pack(fill=tk.X, pady=6)

        tk.Button(ctrl_bar, text="⏮ Prev", command=self.t2s_prev, bg="#3a3f50", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=10, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=5)
        self.t2s_play_btn = tk.Button(ctrl_bar, text="▶ Play Sequence", command=self.t2s_toggle_play, bg="#06d6a0", fg="#1a1c23", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=16, pady=4, cursor="hand2")
        self.t2s_play_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_bar, text="Next ⏭", command=self.t2s_next, bg="#3a3f50", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=10, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_bar, text="Speed (s):", font=("Segoe UI", 8), fg="#9aa0b4", bg="#1a1c23").pack(side=tk.LEFT, padx=(15, 4))
        self.t2s_speed_slider = tk.Scale(ctrl_bar, from_=0.4, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, bg="#1a1c23", fg="white", highlightthickness=0, command=lambda v: setattr(self, 't2s_speed', float(v)))
        self.t2s_speed_slider.set(0.9)
        self.t2s_speed_slider.pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(ctrl_bar, text="Speak audio narration", variable=self.t2s_speak_audio, bg="#1a1c23", fg="#e0e6ed", selectcolor="#20232c", activebackground="#1a1c23", font=("Segoe UI", 8)).pack(side=tk.RIGHT, padx=10)

        # Bottom Timeline
        time_frame = tk.LabelFrame(main_frame, text=" 2. Sign Translation Timeline ", font=("Segoe UI", 9, "bold"), fg="#8d99ae", bg="#20232c", bd=1, relief=tk.SOLID)
        time_frame.pack(fill=tk.X, pady=(6, 0))

        self.t2s_timeline_label = tk.Label(time_frame, text="", font=("Segoe UI", 10), fg="#00e5ff", bg="#20232c", wraplength=1040, justify=tk.LEFT)
        self.t2s_timeline_label.pack(fill=tk.X, padx=12, pady=6)

    def _build_status_bar(self):
        status_bar = tk.Frame(self, bg="#121318", height=24)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(status_bar, text="Bi-Directional Communication Suite Ready.", font=("Segoe UI", 8), fg="#8d99ae", bg="#121318")
        self.status_label.pack(side=tk.LEFT, padx=15, pady=3)

    # =========================================================================
    # TAB 1 LOGIC (SIGN ➔ AUDIO)
    # =========================================================================
    def start_camera(self):
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_camera_running = True
                self.toggle_cam_btn.config(text="Stop Camera", bg="#d90429")
                self.update_video_frame()

    def toggle_camera(self):
        if self.is_camera_running:
            self.is_camera_running = False
            if self.cap:
                self.cap.release()
            self.toggle_cam_btn.config(text="Start Camera", bg="#2b9348")
            self.video_label.config(image="", text="Camera Paused", fg="#8d99ae", font=("Segoe UI", 14))
        else:
            self.start_camera()

    def update_video_frame(self):
        if not self.is_camera_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            annotated_frame, results = self.detector.find_hands(frame, draw=True)
            hands_data = self.detector.get_hand_landmarks_list(annotated_frame, results)

            if hands_data:
                primary = hands_data[0]
                features = self.detector.extract_normalized_features(primary['raw_landmarks'])
                sign, conf, source = self.classifier.classify(features, primary['landmarks'], primary['type'])
                confirmed, progress, _ = self.sentence_builder.update(sign, conf)

                self.sign_display.config(text=sign, fg="#06d6a0" if sign != "UNKNOWN" else "#999999")
                self.conf_display.config(text=f"Confidence: {int(conf * 100)}% | Engine: {source}")
                self.progress_bar['value'] = int(progress * 100)

                if confirmed:
                    self.sentence_entry.delete(0, tk.END)
                    self.sentence_entry.insert(0, self.sentence_builder.sentence)
                    if self.auto_speak_var.get():
                        self.tts.speak(confirmed)
            else:
                self.sign_display.config(text="NO HAND", fg="#4cc9f0")
                self.conf_display.config(text="Confidence: 0% | Standby")
                self.progress_bar['value'] = 0
                self.sentence_builder.update("UNKNOWN", 0.0)

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_frame.shape
            target_w = 600
            target_h = int(h * (target_w / w))
            resized = cv2.resize(rgb_frame, (target_w, target_h))

            img_pil = Image.fromarray(resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.img_tk = img_tk
            self.video_label.config(image=img_tk)

        self.after(20, self.update_video_frame)

    def speak_current_sentence(self):
        text = self.sentence_entry.get().strip()
        if text:
            self.tts.speak(text)
            self.history_box.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
            self.history_box.see(tk.END)
            self.status_label.config(text=f"Spoken: \"{text}\"")

    def add_space(self):
        self.sentence_builder.add_space()
        self.sentence_entry.delete(0, tk.END)
        self.sentence_entry.insert(0, self.sentence_builder.sentence)

    def delete_last(self):
        self.sentence_builder.backspace()
        self.sentence_entry.delete(0, tk.END)
        self.sentence_entry.insert(0, self.sentence_builder.sentence)

    def clear_sentence(self):
        self.sentence_builder.clear()
        self.sentence_entry.delete(0, tk.END)

    def on_voice_changed(self, event=None):
        selected_idx = [v['name'] for v in self.tts.available_voices].index(self.voice_var.get())
        self.tts.set_voice(selected_idx)

    # =========================================================================
    # TAB 2 LOGIC (TEXT/MIC ➔ SIGN)
    # =========================================================================
    def t2s_load_phrase(self, text: str):
        self.t2s_entry.delete(0, tk.END)
        self.t2s_entry.insert(0, text)
        self.t2s_translate_and_load(text)

    def t2s_translate_click(self):
        txt = self.t2s_entry.get().strip()
        if txt:
            self.t2s_translate_and_load(txt)

    def t2s_translate_and_load(self, text: str):
        self.t2s_is_playing = False
        self.t2s_play_btn.config(text="▶ Play Sequence", bg="#06d6a0")
        self.t2s_tokens = self.t2s_engine.translate_text(text)
        self.t2s_current_index = 0

        if not self.t2s_tokens:
            self.t2s_timeline_label.config(text="No recognizable sign tokens found.")
            return

        self.t2s_display_current_sign()
        if self.t2s_speak_audio.get():
            self.tts.speak(text)

    def t2s_display_current_sign(self):
        if not self.t2s_tokens or not (0 <= self.t2s_current_index < len(self.t2s_tokens)):
            return

        token_name = self.t2s_tokens[self.t2s_current_index]['token']
        bgr = self.t2s_engine.render_sign_image(token_name, width=420, height=360)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(image=pil)

        self.t2s_canvas.img_tk = img_tk
        self.t2s_canvas.config(image=img_tk)

        parts = []
        for idx, t in enumerate(self.t2s_tokens):
            if idx == self.t2s_current_index:
                parts.append(f"⭐[{t['token']}]⭐")
            else:
                parts.append(f"[{t['token']}]")
        self.t2s_timeline_label.config(text="  ➔  ".join(parts))

    def t2s_toggle_play(self):
        if self.t2s_is_playing:
            self.t2s_is_playing = False
            self.t2s_play_btn.config(text="▶ Play Sequence", bg="#06d6a0")
        else:
            self.t2s_is_playing = True
            self.t2s_play_btn.config(text="⏸ Pause", bg="#f72585")
            threading.Thread(target=self._t2s_play_loop, daemon=True).start()

    def _t2s_play_loop(self):
        while self.t2s_is_playing and self.t2s_tokens:
            self.t2s_display_current_sign()
            time.sleep(self.t2s_speed)

            if not self.t2s_is_playing:
                break

            self.t2s_current_index += 1
            if self.t2s_current_index >= len(self.t2s_tokens):
                self.t2s_current_index = 0
                self.t2s_is_playing = False
                self.after(0, lambda: self.t2s_play_btn.config(text="▶ Replay Sequence", bg="#06d6a0"))
                break

    def t2s_prev(self):
        self.t2s_is_playing = False
        self.t2s_play_btn.config(text="▶ Play Sequence", bg="#06d6a0")
        if self.t2s_tokens:
            self.t2s_current_index = (self.t2s_current_index - 1) % len(self.t2s_tokens)
            self.t2s_display_current_sign()

    def t2s_next(self):
        self.t2s_is_playing = False
        self.t2s_play_btn.config(text="▶ Play Sequence", bg="#06d6a0")
        if self.t2s_tokens:
            self.t2s_current_index = (self.t2s_current_index + 1) % len(self.t2s_tokens)
            self.t2s_display_current_sign()

    def t2s_start_mic(self):
        self.t2s_mic_btn.config(text="🎙️ Listening...", bg="#e63946")

        def worker():
            transcribed = self.t2s_engine.speech_to_text(timeout=5)
            self.after(0, lambda: self._t2s_handle_mic_result(transcribed))

        threading.Thread(target=worker, daemon=True).start()

    def _t2s_handle_mic_result(self, result: str):
        self.t2s_mic_btn.config(text="🎙️ Speak Mic", bg="#7209b7")
        if result.startswith("[Error]") or result.startswith("[Warning]"):
            messagebox.showwarning("Voice Input", result)
        else:
            self.t2s_entry.delete(0, tk.END)
            self.t2s_entry.insert(0, result)
            self.t2s_translate_and_load(result)

    def on_close(self):
        self.is_camera_running = False
        self.t2s_is_playing = False
        if self.cap:
            self.cap.release()
        self.tts.stop()
        self.destroy()


def main():
    app = BiDirectionalSignApp()
    app.mainloop()


if __name__ == "__main__":
    main()
