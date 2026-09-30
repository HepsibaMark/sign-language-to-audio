"""
Hand Detector & Landmark Extractor Module
Uses Google MediaPipe to track 21 3D hand keypoints in real time,
normalizing spatial coordinates for position and scale invariance.
"""

import math
import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """Wrapper around MediaPipe Hands for robust landmark detection."""

    def __init__(self,
                 mode: bool = False,
                 max_hands: int = 2,
                 detection_con: float = 0.7,
                 track_con: float = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # Landmark indices for finger tips and MCP joints
        self.TIP_IDS = [4, 8, 12, 16, 20]      # Thumb, Index, Middle, Ring, Pinky
        self.PIP_IDS = [3, 6, 10, 14, 18]
        self.MCP_IDS = [2, 5, 9, 13, 17]

    def find_hands(self, img: np.ndarray, draw: bool = True):
        """
        Processes image frame and finds all visible hands.
        Returns:
            annotated_img: Image with landmarks drawn (if draw=True)
            results: MediaPipe Process results object
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=2)
                )

        return img, results

    def get_hand_landmarks_list(self, img: np.ndarray, results) -> list:
        """
        Extracts pixel coordinate bounding boxes and landmarks for each detected hand.
        """
        hands_data = []
        h, w, _ = img.shape

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_type = "Right"
                if results.multi_handedness and len(results.multi_handedness) > i:
                    hand_type = results.multi_handedness[i].classification[0].label

                lm_list = []
                x_list, y_list = [], []
                for id_num, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_list.append(cx)
                    y_list.append(cy)
                    lm_list.append({
                        'id': id_num,
                        'x': cx,
                        'y': cy,
                        'z': lm.z,
                        'norm_x': lm.x,
                        'norm_y': lm.y,
                        'norm_z': lm.z
                    })

                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

                hands_data.append({
                    'type': hand_type,
                    'landmarks': lm_list,
                    'raw_landmarks': hand_landmarks,
                    'bbox': bbox
                })

        return hands_data

    @staticmethod
    def extract_normalized_features(hand_landmarks) -> np.ndarray:
        """
        Normalizes 21 3D points relative to wrist (point 0)
        and scales by max span to achieve translation & scale invariance.
        Returns:
            1D numpy array of 63 normalized coordinates [x0, y0, z0, ..., x20, y20, z20]
        """
        coords = []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        coords = np.array(coords, dtype=np.float32)

        # 1. Position Invariance: Shift origin to wrist (point 0)
        wrist = coords[0]
        coords = coords - wrist

        # 2. Scale Invariance: Scale by Euclidean distance from wrist to Middle Finger MCP (point 9)
        ref_dist = np.linalg.norm(coords[9] - coords[0])
        if ref_dist > 1e-4:
            coords = coords / ref_dist
        else:
            max_val = np.max(np.abs(coords))
            if max_val > 1e-4:
                coords = coords / max_val

        return coords.flatten()

    def get_finger_extension_states(self, lm_list: list, hand_type: str = "Right") -> list:
        """
        Returns boolean list [Thumb, Index, Middle, Ring, Pinky]:
        True = Extended / Open, False = Folded / Closed.
        """
        if len(lm_list) < 21:
            return [False, False, False, False, False]

        fingers = []

        # Thumb: compare tip (4) with IP joint (3) along horizontal axis depending on hand
        if hand_type == "Right":
            fingers.append(lm_list[self.TIP_IDS[0]]['x'] < lm_list[self.TIP_IDS[0] - 1]['x'])
        else:
            fingers.append(lm_list[self.TIP_IDS[0]]['x'] > lm_list[self.TIP_IDS[0] - 1]['x'])

        # 4 Fingers: tip Y is higher (smaller pixel value) than PIP joint Y
        for id_idx in range(1, 5):
            tip_id = self.TIP_IDS[id_idx]
            pip_id = self.PIP_IDS[id_idx]
            fingers.append(lm_list[tip_id]['y'] < lm_list[pip_id]['y'])

        return fingers

    @staticmethod
    def calculate_distance(p1: dict, p2: dict) -> float:
        """Euclidean distance between two landmarks in pixel space."""
        return math.hypot(p2['x'] - p1['x'], p2['y'] - p1['y'])
