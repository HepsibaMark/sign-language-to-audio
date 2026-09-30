"""
Interactive Gesture Data Collector
Allows users to easily collect new custom sign language gestures using their webcam.
Captures normalized 21 3D landmarks and appends them to the dataset.
"""

import os
import pickle
import time
import cv2
import numpy as np
from src.hand_detector import HandDetector


def run_collector(num_samples: int = 60):
    dataset_dir = os.path.join(os.path.dirname(__file__), "data")
    dataset_path = os.path.join(dataset_dir, "landmarks_dataset.pickle")
    os.makedirs(dataset_dir, exist_ok=True)

    # Load existing dataset if available
    existing_data = []
    existing_labels = []
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, 'rb') as f:
                d = pickle.load(f)
                existing_data = list(d.get('data', []))
                existing_labels = list(d.get('labels', []))
                print(f"[Collector] Loaded existing dataset with {len(existing_data)} samples.")
        except Exception as e:
            print(f"[Collector] Warning loading existing dataset: {e}")

    print("\n" + "="*60)
    print("  SIGN LANGUAGE DATA COLLECTOR")
    print("="*60)
    gesture_name = input("Enter gesture label to record (e.g. HELLO, A, B, HELP): ").strip().upper()
    if not gesture_name:
        print("[Collector] No gesture name entered. Exiting.")
        return

    print(f"\n[Collector] Preparing to record '{gesture_name}' ({num_samples} frames)...")
    print("Press 'S' on the video window when ready to start capture.")
    print("Press 'Q' anytime to exit.")

    detector = HandDetector(max_hands=1, detection_con=0.7)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Could not open webcam (index 0).")
        return

    # Phase 1: Positioning & Ready state
    ready_to_record = False
    while not ready_to_record:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        annotated, results = detector.find_hands(frame, draw=True)

        # Overlay instructions
        cv2.rectangle(annotated, (10, 10), (w - 10, 70), (20, 20, 20), -1)
        cv2.putText(annotated, f"Target Gesture: {gesture_name}", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated, "Position hand and press 'S' to begin capture", (25, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Sign Language Collector", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):
            ready_to_record = True
        elif key == ord('q') or key == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()
            print("[Collector] Cancelled by user.")
            return

    # Countdown (3, 2, 1)
    for cd in [3, 2, 1]:
        start_t = time.time()
        while time.time() - start_t < 1.0:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            cv2.putText(frame, str(cd), (w // 2 - 30, h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5)
            cv2.imshow("Sign Language Collector", frame)
            cv2.waitKey(1)

    # Phase 2: Capture samples
    collected = 0
    print(f"[Collector] Recording {num_samples} samples...")

    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        annotated, results = detector.find_hands(frame, draw=True)

        if results.multi_hand_landmarks:
            raw_lm = results.multi_hand_landmarks[0]
            features = detector.extract_normalized_features(raw_lm)

            existing_data.append(features)
            existing_labels.append(gesture_name)
            collected += 1

            # Visual progress bar
            bar_w = int((collected / num_samples) * (w - 60))
            cv2.rectangle(annotated, (30, h - 50), (w - 30, h - 25), (40, 40, 40), -1)
            cv2.rectangle(annotated, (30, h - 50), (30 + bar_w, h - 25), (0, 255, 0), -1)
            cv2.putText(annotated, f"Recording: {collected}/{num_samples} frames", (35, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "NO HAND DETECTED! Place hand in view", (30, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Sign Language Collector", annotated)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected > 0:
        with open(dataset_path, 'wb') as f:
            pickle.dump({'data': np.array(existing_data), 'labels': np.array(existing_labels)}, f)
        print(f"\n[Collector] Success! Saved {collected} new samples for '{gesture_name}'.")
        print(f"Total dataset samples: {len(existing_data)}.")
        print("Run 'python train_model.py' to retrain your model with the new data!")
    else:
        print("[Collector] No samples were captured.")


if __name__ == "__main__":
    run_collector()
