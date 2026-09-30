"""
Sign Language to Audio - Main OpenCV HUD Application
Real-time webcam gesture detection, sentence building, and audio synthesis.
"""

import time
import cv2
import numpy as np
from src.hand_detector import HandDetector
from src.gesture_classifier import GestureClassifier
from src.sentence_builder import SentenceBuilder
from src.tts_engine import TTSEngine


def draw_hud(img: np.ndarray,
             current_sign: str,
             confidence: float,
             source: str,
             hold_progress: float,
             sentence: str,
             is_speaking: bool,
             auto_speak: bool,
             fps: float):
    """Draws an informative, modern HUD overlay on the video feed."""
    h, w, _ = img.shape

    # 1. Top Header Banner
    header_h = 55
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # Title & FPS
    cv2.putText(img, "SIGN LANGUAGE TO AUDIO", (20, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 240, 255), 2)
    mode_str = "AUTO-SPEAK: ON" if auto_speak else "AUTO-SPEAK: OFF (Press M)"
    cv2.putText(img, mode_str, (w - 380, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180) if auto_speak else (180, 180, 180), 1)
    cv2.putText(img, f"FPS: {fps:.1f}", (w - 110, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # 2. Left Gesture Card (Detection & Confidence)
    card_w = 260
    card_h = 130
    cx, cy = 20, 75
    cv2.rectangle(img, (cx, cy), (cx + card_w, cy + card_h), (20, 20, 20), -1)
    cv2.rectangle(img, (cx, cy), (cx + card_w, cy + card_h), (60, 60, 60), 1)

    cv2.putText(img, "DETECTED GESTURE", (cx + 15, cy + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Sign label color
    sign_color = (0, 255, 0) if current_sign != "UNKNOWN" else (100, 100, 100)
    cv2.putText(img, current_sign, (cx + 15, cy + 62),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, sign_color, 2)

    # Confidence & Source
    conf_pct = int(confidence * 100)
    cv2.putText(img, f"Conf: {conf_pct}% ({source})", (cx + 15, cy + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Hold Progress Bar
    bar_x, bar_y = cx + 15, cy + 104
    bar_max_w = card_w - 30
    bar_fill_w = int(hold_progress * bar_max_w)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 12), (50, 50, 50), -1)
    if bar_fill_w > 0:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_fill_w, bar_y + 12), (0, 220, 255), -1)

    # 3. Bottom Sentence & Audio Dashboard
    bottom_h = 110
    by = h - bottom_h
    overlay2 = img.copy()
    cv2.rectangle(overlay2, (0, by), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.90, img, 0.10, 0, img)

    # Audio synthesis indicator
    audio_icon_color = (0, 255, 0) if is_speaking else (120, 120, 120)
    audio_text = "[AUDIO SPEAKING...]" if is_speaking else "[AUDIO READY]"
    cv2.circle(img, (30, by + 28), 8, audio_icon_color, -1)
    cv2.putText(img, audio_text, (50, by + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, audio_icon_color, 2)

    # Sentence Banner
    display_sent = sentence if sentence else "(Hold gesture to build sentence...)"
    sent_color = (255, 255, 255) if sentence else (140, 140, 140)
    cv2.putText(img, "SENTENCE:", (30, by + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(img, f'"{display_sent}"', (140, by + 68),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, sent_color, 2)

    # Help hotkey guide
    cv2.putText(img, "[ENTER/S] Speak  |  [SPACE] Space  |  [BKSP] Delete  |  [C] Clear  |  [M] Mode  |  [Q] Exit",
                (30, by + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

    return img


def run_app():
    print("=" * 60)
    print("  SIGN LANGUAGE TO AUDIO - REAL-TIME ASSISTIVE SYSTEM")
    print("=" * 60)

    # Initialize modules
    print("[System] Initializing Hand Detector (MediaPipe)...")
    detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.5)

    print("[System] Initializing Gesture Classifier...")
    classifier = GestureClassifier()

    print("[System] Initializing Sentence Builder...")
    sentence_builder = SentenceBuilder(hold_time_seconds=0.8, cooldown_seconds=0.6)

    print("[System] Initializing Text-to-Speech Engine...")
    tts = TTSEngine(rate=160, volume=1.0)
    tts.speak("Sign language to audio system is ready.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not access webcam. Ensure a camera is connected.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    auto_speak = False
    current_sign = "UNKNOWN"
    current_conf = 0.0
    current_source = "None"
    hold_progress = 0.0

    print("\n[Application Running] Press 'Q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab frame from camera.")
            break

        # Flip horizontally for natural mirror interaction
        frame = cv2.flip(frame, 1)

        # FPS computation
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        # Hand detection
        annotated_frame, results = detector.find_hands(frame, draw=True)
        hands_data = detector.get_hand_landmarks_list(annotated_frame, results)

        if hands_data:
            primary_hand = hands_data[0]
            raw_lm = primary_hand['raw_landmarks']
            lm_list = primary_hand['landmarks']
            hand_type = primary_hand['type']

            # Extract normalized 63 features
            features = detector.extract_normalized_features(raw_lm)

            # Classify
            sign_label, conf, source = classifier.classify(features, lm_list, hand_type)
            current_sign = sign_label
            current_conf = conf
            current_source = source

            # Sentence buffer & stabilization update
            confirmed, progress, in_cooldown = sentence_builder.update(sign_label, conf)
            hold_progress = progress

            if confirmed:
                print(f"[Confirmed Sign] '{confirmed}' added to sentence.")
                if auto_speak:
                    tts.speak(confirmed)
        else:
            current_sign = "NO HAND"
            current_conf = 0.0
            current_source = "None"
            hold_progress = 0.0
            sentence_builder.update("UNKNOWN", 0.0)

        # Render HUD Overlay
        output_frame = draw_hud(
            annotated_frame,
            current_sign=current_sign,
            confidence=current_conf,
            source=current_source,
            hold_progress=hold_progress,
            sentence=sentence_builder.sentence,
            is_speaking=tts.is_speaking,
            auto_speak=auto_speak,
            fps=fps
        )

        cv2.imshow("Sign Language to Audio Assistive System", output_frame)
        key = cv2.waitKey(1) & 0xFF

        # Handle keyboard interactions
        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            break
        elif key in [13, ord('s'), ord('S')]:  # ENTER or 'S' -> Speak
            sentence = sentence_builder.sentence.strip()
            if sentence:
                print(f"[Speech Action] Speaking sentence: \"{sentence}\"")
                tts.speak(sentence)
        elif key == 32:  # SPACE
            sentence_builder.add_space()
        elif key == 8:   # BACKSPACE
            sentence_builder.backspace()
        elif key == ord('c') or key == ord('C'):  # Clear
            sentence_builder.clear()
        elif key == ord('m') or key == ord('M'):  # Toggle Auto-speak
            auto_speak = not auto_speak
            print(f"[Settings] Auto-speak toggled: {auto_speak}")

    cap.release()
    cv2.destroyAllWindows()
    tts.stop()
    print("[System] Application shut down cleanly.")


if __name__ == "__main__":
    run_app()
