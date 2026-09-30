"""
Gesture Classifier Module
Combines Machine Learning (Random Forest / MLP) with a robust geometric
heuristic engine to recognize American Sign Language (ASL) and essential everyday phrases.
"""

import math
import os
import pickle
import numpy as np


class GestureClassifier:
    """Classifies hand gestures into letters, words, and conversational signs."""

    def __init__(self, model_path: str = None):
        self.model = None
        self.classes = []
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "..", "model", "gesture_model.p")
        self.load_model()

    def load_model(self):
        """Loads trained scikit-learn model if present."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.classes = data.get('classes', [])
                print(f"[Classifier] Loaded trained model from {self.model_path} with {len(self.classes)} classes.")
            except Exception as e:
                print(f"[Classifier] Note: Could not load model ({e}). Using geometric engine.")
                self.model = None
        else:
            self.model = None

    def classify(self, normalized_features: np.ndarray, lm_list: list, hand_type: str = "Right") -> tuple:
        """
        Classifies current hand pose.
        Returns:
            (predicted_label: str, confidence: float, source: str)
        """
        # 1. Try ML model prediction if available
        if self.model is not None and normalized_features is not None:
            try:
                features = normalized_features.reshape(1, -1)
                probabilities = self.model.predict_proba(features)[0]
                max_idx = np.argmax(probabilities)
                conf = float(probabilities[max_idx])
                predicted_label = self.classes[max_idx]

                if conf >= 0.70:
                    return predicted_label, conf, "ML"
            except Exception as e:
                pass

        # 2. Geometric Heuristic Engine (Rule-based expert system)
        heuristic_label, conf = self.classify_geometric(lm_list, hand_type)
        return heuristic_label, conf, "Geometric"

    def classify_geometric(self, lm: list, hand_type: str = "Right") -> tuple:
        """
        Calculates geometric angles and finger configurations to identify
        common ASL letters and essential social phrases.
        """
        if not lm or len(lm) < 21:
            return "UNKNOWN", 0.0

        # Extract landmarks (pixel coordinates)
        # 0: Wrist
        # 4: Thumb Tip, 3: Thumb IP, 2: Thumb MCP
        # 8: Index Tip, 7: Index DIP, 6: Index PIP, 5: Index MCP
        # 12: Middle Tip, 11: Middle DIP, 10: Middle PIP, 9: Middle MCP
        # 16: Ring Tip, 15: Ring DIP, 14: Ring PIP, 13: Ring MCP
        # 20: Pinky Tip, 19: Pinky DIP, 18: Pinky PIP, 17: Pinky MCP

        def dist(i, j):
            return math.hypot(lm[i]['x'] - lm[j]['x'], lm[i]['y'] - lm[j]['y'])

        # Reference scale: wrist (0) to middle MCP (9)
        ref_scale = dist(0, 9)
        if ref_scale == 0:
            return "UNKNOWN", 0.0

        def norm_dist(i, j):
            return dist(i, j) / ref_scale

        # Finger extension states (True = extended, False = folded)
        # For fingers 1..4: tip.y is significantly higher (smaller value) than pip.y
        index_open = lm[8]['y'] < lm[6]['y'] - (0.1 * ref_scale)
        middle_open = lm[12]['y'] < lm[10]['y'] - (0.1 * ref_scale)
        ring_open = lm[16]['y'] < lm[14]['y'] - (0.1 * ref_scale)
        pinky_open = lm[20]['y'] < lm[18]['y'] - (0.1 * ref_scale)

        # Thumb extension: tip distance from index MCP (5)
        thumb_extended = norm_dist(4, 5) > 0.65
        thumb_up = lm[4]['y'] < lm[3]['y'] and lm[3]['y'] < lm[2]['y']
        thumb_down = lm[4]['y'] > lm[3]['y'] and lm[3]['y'] > lm[2]['y']

        # Inter-finger distances
        thumb_index_dist = norm_dist(4, 8)
        index_middle_dist = norm_dist(8, 12)
        middle_ring_dist = norm_dist(12, 16)
        thumb_middle_dist = norm_dist(4, 12)

        # ------------------- Everyday & Social Phrases -------------------

        # 1. "I LOVE YOU" (Thumb + Index + Pinky OPEN, Middle + Ring CLOSED)
        if thumb_extended and index_open and pinky_open and not middle_open and not ring_open:
            return "I LOVE YOU", 0.96

        # 2. "HELLO" / "OPEN PALM" (All 5 fingers open and extended)
        if thumb_extended and index_open and middle_open and ring_open and pinky_open:
            return "HELLO", 0.95

        # 3. "HELP" / "THUMBS UP" / "GOOD" (Thumb up, all 4 fingers curled)
        if thumb_up and not index_open and not middle_open and not ring_open and not pinky_open:
            # Check if thumb points distinctly upward
            if lm[4]['y'] < lm[2]['y'] - (0.4 * ref_scale):
                return "HELP / GOOD", 0.94

        # 4. "THUMBS DOWN" / "BAD" (Thumb pointing downwards, other fingers curled)
        if thumb_down and not index_open and not middle_open and not ring_open and not pinky_open:
            if lm[4]['y'] > lm[2]['y'] + (0.4 * ref_scale):
                return "BAD / NO", 0.92

        # 5. "PEACE / VICTORY" or ASL "V" (Index & Middle open separated, Ring & Pinky closed)
        if index_open and middle_open and not ring_open and not pinky_open:
            if index_middle_dist > 0.28:
                return "PEACE (V)", 0.94
            else:
                return "U", 0.92

        # 6. "OK" or ASL "F" (Thumb and Index touching in ring, Middle, Ring, Pinky open)
        if thumb_index_dist < 0.28 and middle_open and ring_open and pinky_open:
            return "OK", 0.95

        # 7. "WATER" or ASL "W" (Index, Middle, Ring open, Pinky closed, Thumb over pinky)
        if index_open and middle_open and ring_open and not pinky_open:
            return "WATER (W)", 0.93

        # 8. ASL "L" (Thumb and Index extended at roughly 90 degrees, others closed)
        if thumb_extended and index_open and not middle_open and not ring_open and not pinky_open:
            return "L", 0.94

        # 9. ASL "Y" / "PHONE / CALL" (Thumb and Pinky extended, Index, Middle, Ring closed)
        if thumb_extended and pinky_open and not index_open and not middle_open and not ring_open:
            return "CALL / Y", 0.94

        # 10. ASL "I" (Only Pinky open, others closed)
        if pinky_open and not index_open and not middle_open and not ring_open and not thumb_extended:
            return "I", 0.93

        # 11. ASL "D" or Pointing (Index open, Middle, Ring, Pinky closed, Thumb touches Middle)
        if index_open and not middle_open and not ring_open and not pinky_open and not thumb_extended:
            return "YES (D)", 0.91

        # 12. ASL "C" (Curved fingers forming a 'C' shape)
        if not index_open and not middle_open and not ring_open and not pinky_open:
            # Check if thumb is separated but tips are forward
            if 0.35 < thumb_index_dist < 0.65 and lm[8]['y'] > lm[6]['y'] - (0.2 * ref_scale):
                return "C", 0.85

        # 13. ASL "A" / "FIST" / "STOP" (Closed fist, thumb resting beside index)
        if not index_open and not middle_open and not ring_open and not pinky_open:
            if lm[4]['y'] < lm[5]['y']:
                return "A", 0.90
            else:
                return "S", 0.88

        # 14. ASL "B" (All 4 fingers straight up, thumb tucked across palm)
        if index_open and middle_open and ring_open and pinky_open and not thumb_extended:
            return "THANK YOU (B)", 0.92

        return "UNKNOWN", 0.40
