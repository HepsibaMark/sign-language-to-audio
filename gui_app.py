"""
Sign Language to Audio - Desktop GUI Application
A full-featured Tkinter desktop application with live camera integration,
audio voice/speed controls, sentence constructor, and history logging.
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


class SignLanguageApp(tk.Tk):
    """Tkinter-based GUI for the Sign Language to Audio assistive system."""

    def __init__(self):
        super().__init__()

        self.title("Sign Language to Audio - Assistive Communication Suite")
        self.geometry("1180x760")
        self.minsize(1000, 680)
        self.configure(bg="#1e1e24")

        # Core logic components
        self.detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.5)
        self.classifier = GestureClassifier()
        self.sentence_builder = SentenceBuilder(hold_time_seconds=0.8, cooldown_seconds=0.6)
        self.tts = TTSEngine(rate=160, volume=1.0)

        # Video capture state
        self.cap = None
        self.is_camera_running = False
        self.auto_speak_var = tk.BooleanVar(value=False)

        # Build UI layout
        self._build_header()
        self._build_main_content()
        self._build_status_bar()

        # Start camera loop
        self.start_camera()

        # Protocol for clean window exit
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_header(self):
        header = tk.Frame(self, bg="#2b2d42", height=60)
        header.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(
            header,
            text="SIGN LANGUAGE TO AUDIO ASSISTIVE SYSTEM",
            font=("Segoe UI", 16, "bold"),
            fg="#edf2f4",
            bg="#2b2d42"
        )
        title.pack(side=tk.LEFT, padx=20, pady=12)

        subtitle = tk.Label(
            header,
            text="Empowering Inclusive Communication with AI & Speech Synthesis",
            font=("Segoe UI", 10, "italic"),
            fg="#8d99ae",
            bg="#2b2d42"
        )
        subtitle.pack(side=tk.LEFT, padx=5, pady=16)

    def _build_main_content(self):
        main_frame = tk.Frame(self, bg="#1e1e24")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left Column: Video Feed
        left_col = tk.Frame(main_frame, bg="#1e1e24")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        video_card = tk.LabelFrame(
            left_col,
            text=" Live Video Camera Feed ",
            font=("Segoe UI", 11, "bold"),
            fg="#8d99ae",
            bg="#252733",
            bd=1,
            relief=tk.SOLID
        )
        video_card.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(video_card, bg="#0d0e12")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Video control buttons
        vid_btn_bar = tk.Frame(video_card, bg="#252733")
        vid_btn_bar.pack(fill=tk.X, pady=(0, 8), padx=8)

        self.toggle_cam_btn = tk.Button(
            vid_btn_bar,
            text="Stop Camera",
            command=self.toggle_camera,
            bg="#d90429",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2"
        )
        self.toggle_cam_btn.pack(side=tk.LEFT, padx=5)

        auto_speak_chk = tk.Checkbutton(
            vid_btn_bar,
            text="Auto-speak words on confirmation",
            variable=self.auto_speak_var,
            bg="#252733",
            fg="#edf2f4",
            selectcolor="#1e1e24",
            activebackground="#252733",
            activeforeground="#edf2f4",
            font=("Segoe UI", 9)
        )
        auto_speak_chk.pack(side=tk.LEFT, padx=15)

        # Right Column: Controls, Sentence Buffer, Speech Settings
        right_col = tk.Frame(main_frame, bg="#1e1e24", width=420)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_col.pack_propagate(False)

        # Card 1: Gesture Detection Status
        det_card = tk.LabelFrame(
            right_col,
            text=" Recognized Gesture ",
            font=("Segoe UI", 11, "bold"),
            fg="#8d99ae",
            bg="#252733",
            bd=1,
            relief=tk.SOLID
        )
        det_card.pack(fill=tk.X, pady=(0, 10))

        self.sign_display = tk.Label(
            det_card,
            text="NO HAND",
            font=("Segoe UI", 26, "bold"),
            fg="#4cc9f0",
            bg="#252733"
        )
        self.sign_display.pack(pady=(8, 2))

        self.conf_display = tk.Label(
            det_card,
            text="Confidence: 0% | Model: None",
            font=("Segoe UI", 9),
            fg="#a0aab2",
            bg="#252733"
        )
        self.conf_display.pack(pady=(0, 4))

        self.progress_bar = ttk.Progressbar(det_card, orient=tk.HORIZONTAL, length=380, mode='determinate')
        self.progress_bar.pack(pady=(2, 10), padx=15)

        # Card 2: Sentence Builder & Speech Actions
        sent_card = tk.LabelFrame(
            right_col,
            text=" Sentence Builder ",
            font=("Segoe UI", 11, "bold"),
            fg="#8d99ae",
            bg="#252733",
            bd=1,
            relief=tk.SOLID
        )
        sent_card.pack(fill=tk.X, pady=(0, 10))

        self.sentence_entry = tk.Entry(
            sent_card,
            font=("Segoe UI", 13),
            bg="#181a20",
            fg="#edf2f4",
            insertbackground="white",
            bd=1,
            relief=tk.SOLID
        )
        self.sentence_entry.pack(fill=tk.X, padx=12, pady=(10, 8))

        btn_row1 = tk.Frame(sent_card, bg="#252733")
        btn_row1.pack(fill=tk.X, padx=12, pady=4)

        speak_btn = tk.Button(
            btn_row1,
            text="🔊 Speak Sentence",
            command=self.speak_current_sentence,
            bg="#4361ee",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2"
        )
        speak_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        space_btn = tk.Button(
            btn_row1,
            text="Space",
            command=self.add_space,
            bg="#3a0ca3",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=6,
            cursor="hand2"
        )
        space_btn.pack(side=tk.LEFT, padx=4)

        bksp_btn = tk.Button(
            btn_row1,
            text="Delete",
            command=self.delete_last,
            bg="#480ca8",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=6,
            cursor="hand2"
        )
        bksp_btn.pack(side=tk.LEFT, padx=4)

        clear_btn = tk.Button(
            btn_row1,
            text="Clear",
            command=self.clear_sentence,
            bg="#560bad",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=6,
            cursor="hand2"
        )
        clear_btn.pack(side=tk.LEFT, padx=(4, 0))

        # Card 3: Audio Settings & History
        settings_card = tk.LabelFrame(
            right_col,
            text=" Voice Controls & Spoken History ",
            font=("Segoe UI", 11, "bold"),
            fg="#8d99ae",
            bg="#252733",
            bd=1,
            relief=tk.SOLID
        )
        settings_card.pack(fill=tk.BOTH, expand=True)

        voice_row = tk.Frame(settings_card, bg="#252733")
        voice_row.pack(fill=tk.X, padx=12, pady=(8, 4))

        tk.Label(voice_row, text="Voice:", font=("Segoe UI", 9), fg="#edf2f4", bg="#252733").pack(side=tk.LEFT)
        voice_names = [v['name'] for v in self.tts.available_voices] or ["Default"]
        self.voice_var = tk.StringVar(value=voice_names[0])
        voice_dropdown = ttk.Combobox(voice_row, textvariable=self.voice_var, values=voice_names, state="readonly", width=32)
        voice_dropdown.pack(side=tk.LEFT, padx=8)
        voice_dropdown.bind("<<ComboboxSelected>>", self.on_voice_changed)

        slider_row = tk.Frame(settings_card, bg="#252733")
        slider_row.pack(fill=tk.X, padx=12, pady=4)

        tk.Label(slider_row, text="Speed:", font=("Segoe UI", 9), fg="#edf2f4", bg="#252733").pack(side=tk.LEFT)
        self.speed_slider = tk.Scale(slider_row, from_=100, to=240, orient=tk.HORIZONTAL, bg="#252733", fg="white",
                                     highlightthickness=0, command=self.on_speed_changed)
        self.speed_slider.set(160)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        tk.Label(settings_card, text="Speech Log:", font=("Segoe UI", 9, "bold"), fg="#8d99ae", bg="#252733").pack(anchor="w", padx=12, pady=(4, 2))
        self.history_box = scrolledtext.ScrolledText(settings_card, height=6, bg="#181a20", fg="#4cc9f0", font=("Consolas", 9), bd=1, relief=tk.SOLID)
        self.history_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

    def _build_status_bar(self):
        status_bar = tk.Frame(self, bg="#121318", height=24)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(
            status_bar,
            text="System Initialized. Hold gesture for 0.8s to register.",
            font=("Segoe UI", 8),
            fg="#8d99ae",
            bg="#121318"
        )
        self.status_label.pack(side=tk.LEFT, padx=15, pady=3)

    def start_camera(self):
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_camera_running = True
                self.toggle_cam_btn.config(text="Stop Camera", bg="#d90429")
                self.update_video_frame()
            else:
                messagebox.showerror("Camera Error", "Could not connect to webcam. Please ensure a camera is available.")

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
                self.conf_display.config(text="Confidence: 0% | Engine: Standby")
                self.progress_bar['value'] = 0
                self.sentence_builder.update("UNKNOWN", 0.0)

            # Resize frame to fit GUI video container
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_frame.shape
            target_w = 640
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

    def on_speed_changed(self, val):
        self.tts.set_rate(int(val))

    def on_close(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        self.tts.stop()
        self.destroy()


def main():
    app = SignLanguageApp()
    app.mainloop()


if __name__ == "__main__":
    main()
