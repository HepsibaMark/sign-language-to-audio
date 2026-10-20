"""
Text to Sign Language (T2SL) & Speech to Sign Language Engine
Translates written text or spoken voice into American Sign Language (ASL)
fingerspelling sequences and animated whole-phrase gesture frames.
"""

import os
import re
import math
import cv2
import numpy as np

# Landmark joint connections matching MediaPipe Hands standard
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0),
    # Thumb
    (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20)
]

# Color palette for finger bone segments (BGR format)
FINGER_COLORS = {
    'palm': (220, 220, 220),
    'thumb': (0, 180, 255),    # Vibrant Orange
    'index': (0, 240, 255),    # Bright Yellow
    'middle': (0, 255, 120),   # Neon Green
    'ring': (255, 160, 0),     # Azure Blue
    'pinky': (255, 50, 200)    # Hot Pink
}


class TextToSignEngine:
    """Translates text/speech sentences into sequential Sign Language representations."""

    def __init__(self, assets_dir: str = None):
        self.assets_dir = assets_dir or os.path.join(os.path.dirname(__file__), "..", "assets", "signs")
        os.makedirs(self.assets_dir, exist_ok=True)
        self.cached_images = {}
        self.dictionary_descriptions = self._get_descriptions()

    def _get_descriptions(self) -> dict:
        """Sign descriptions for educational and guidance purposes."""
        return {
            'A': "Fist with thumb upright against side of index",
            'B': "Four upright fingers touching flat, thumb folded in",
            'C': "All fingers curved in a smooth 'C' arc",
            'D': "Index finger pointing up, others touching thumb in ring",
            'E': "Fingers bent downward at knuckles, thumb tucked below",
            'F': "Thumb and index tips touching in ring, others upright",
            'G': "Index finger and thumb pointing forward horizontally",
            'H': "Index and middle fingers extended together horizontally",
            'I': "Only pinky finger extended straight up",
            'J': "Pinky finger drawing a 'J' curve in the air",
            'K': "Index pointing up, middle forward, thumb between them",
            'L': "Thumb and index extended at right angle (90 degrees)",
            'M': "Three fingers draped over thumb folded into palm",
            'N': "Two fingers draped over thumb folded into palm",
            'O': "All five fingertips touching thumb tip to form an 'O'",
            'P': "Index finger pointing downward with middle angled",
            'Q': "Thumb and index pointing downwards like G",
            'R': "Index and middle fingers crossed over each other",
            'S': "Tight closed fist with thumb wrapped over fingers",
            'T': "Index finger tucked over thumb in fist",
            'U': "Index and middle fingers held straight up and touching",
            'V': "Index and middle fingers separated in a 'V' peace shape",
            'W': "Index, middle, and ring fingers extended in 'W' shape",
            'X': "Index finger curled into a small hook shape",
            'Y': "Thumb and pinky extended outward in 'shaka' sign",
            'Z': "Index finger tracing the letter 'Z' in the air",
            'HELLO': "Open flat palm held forward with fingers spread",
            'THANK YOU': "Flat hand touching chin, moving forward towards viewer",
            'YES': "Closed fist nodding up and down",
            'NO': "Index and middle snapping closed against thumb",
            'HELP': "Thumbs-up resting over open flat palm",
            'WATER': "Three fingers in 'W' tapped gently against mouth/chin",
            'I LOVE YOU': "Thumb, index, and pinky extended simultaneously",
            'OK': "Thumb and index touching in circle, other 3 extended",
            'PEACE': "Index and middle separated in 'V' shape",
            'CALL': "Thumb and pinky held near ear like telephone handset",
            'PLEASE': "Flat palm rubbing in circle over chest",
            'GOOD': "Thumbs-up gesture with firm grip",
            'BAD': "Thumbs-down gesture with firm grip",
            'SPACE': "Pause between words (Hands at rest)"
        }

    def generate_landmark_pose(self, sign_name: str) -> np.ndarray:
        """
        Synthesizes standard 21 3D landmarks for any letter A-Z or common word sign.
        Returns: 21x3 numpy array.
        """
        s = sign_name.upper().strip()
        lm = np.zeros((21, 3), dtype=np.float32)

        # 0: Wrist
        lm[0] = [0.0, 0.0, 0.0]

        # Determine thumb & finger extension states
        thumb_state = "folded"
        index_state = "folded"
        middle_state = "folded"
        ring_state = "folded"
        pinky_state = "folded"
        spread = 0.0
        hook_index = False
        crossed = False

        if s == 'A':
            thumb_state = "side"
        elif s == 'B':
            thumb_state = "folded"
            index_state = middle_state = ring_state = pinky_state = "extended"
        elif s == 'C':
            thumb_state = "arc"
            index_state = middle_state = ring_state = pinky_state = "bent"
        elif s == 'D':
            thumb_state = "ring"
            index_state = "extended"
        elif s == 'E':
            thumb_state = "under"
            index_state = middle_state = ring_state = pinky_state = "bent"
        elif s in ['F', 'OK']:
            thumb_state = "ring"
            middle_state = ring_state = pinky_state = "extended"
            spread = 0.1
        elif s == 'G':
            thumb_state = "forward"
            index_state = "forward"
        elif s == 'H':
            thumb_state = "folded"
            index_state = middle_state = "forward"
        elif s == 'I':
            pinky_state = "extended"
        elif s == 'J':
            pinky_state = "extended"
        elif s == 'K':
            thumb_state = "up"
            index_state = "extended"
            middle_state = "forward"
        elif s == 'L':
            thumb_state = "extended"
            index_state = "extended"
        elif s == 'M':
            thumb_state = "under"
            index_state = middle_state = ring_state = "bent"
        elif s == 'N':
            thumb_state = "under"
            index_state = middle_state = "bent"
        elif s == 'O':
            thumb_state = "ring"
            index_state = middle_state = ring_state = pinky_state = "bent"
        elif s == 'P':
            thumb_state = "up"
            index_state = "down"
            middle_state = "forward"
        elif s == 'Q':
            thumb_state = "forward"
            index_state = "down"
        elif s == 'R':
            index_state = middle_state = "extended"
            crossed = True
        elif s == 'S':
            thumb_state = "over"
        elif s == 'T':
            thumb_state = "between"
            index_state = "bent"
        elif s == 'U':
            index_state = middle_state = "extended"
            spread = -0.05
        elif s in ['V', 'PEACE']:
            index_state = middle_state = "extended"
            spread = 0.22
        elif s in ['W', 'WATER']:
            index_state = middle_state = ring_state = "extended"
            spread = 0.12
        elif s == 'X':
            hook_index = True
        elif s in ['Y', 'CALL']:
            thumb_state = "extended"
            pinky_state = "extended"
        elif s == 'Z':
            index_state = "extended"
        elif s == 'HELLO':
            thumb_state = "extended"
            index_state = middle_state = ring_state = pinky_state = "extended"
            spread = 0.2
        elif s in ['HELP', 'GOOD']:
            thumb_state = "up"
        elif s == 'BAD':
            thumb_state = "down"
        elif s == 'I LOVE YOU':
            thumb_state = "extended"
            index_state = "extended"
            pinky_state = "extended"
            spread = 0.15
        elif s == 'THANK YOU':
            thumb_state = "folded"
            index_state = middle_state = ring_state = pinky_state = "extended"
        elif s == 'YES':
            thumb_state = "side"
        elif s == 'NO':
            thumb_state = "ring"
            index_state = middle_state = "bent"
        elif s == 'SPACE':
            # Resting flat hand
            thumb_state = "extended"
            index_state = middle_state = ring_state = pinky_state = "extended"
            spread = 0.05

        # Build Thumb landmarks
        if thumb_state == "extended":
            lm[1] = [-0.22, -0.15, 0.0]
            lm[2] = [-0.45, -0.28, 0.0]
            lm[3] = [-0.68, -0.42, 0.0]
            lm[4] = [-0.85, -0.55, 0.0]
        elif thumb_state == "up":
            lm[1] = [-0.18, -0.15, 0.0]
            lm[2] = [-0.28, -0.32, 0.0]
            lm[3] = [-0.32, -0.55, 0.0]
            lm[4] = [-0.34, -0.80, 0.0]
        elif thumb_state == "down":
            lm[1] = [-0.18, 0.10, 0.0]
            lm[2] = [-0.28, 0.25, 0.0]
            lm[3] = [-0.32, 0.45, 0.0]
            lm[4] = [-0.34, 0.70, 0.0]
        elif thumb_state == "ring":
            lm[1] = [-0.15, -0.15, 0.0]
            lm[2] = [-0.24, -0.32, 0.0]
            lm[3] = [-0.20, -0.50, 0.05]
            lm[4] = [-0.10, -0.62, 0.05]
        elif thumb_state == "side":
            lm[1] = [-0.15, -0.15, 0.0]
            lm[2] = [-0.28, -0.30, 0.0]
            lm[3] = [-0.28, -0.48, 0.0]
            lm[4] = [-0.26, -0.62, 0.0]
        elif thumb_state == "over":
            lm[1] = [-0.15, -0.15, 0.0]
            lm[2] = [-0.10, -0.32, 0.10]
            lm[3] = [0.05, -0.38, 0.12]
            lm[4] = [0.18, -0.40, 0.12]
        else:  # folded
            lm[1] = [-0.15, -0.15, 0.0]
            lm[2] = [-0.24, -0.28, 0.0]
            lm[3] = [-0.18, -0.40, 0.06]
            lm[4] = [-0.10, -0.46, 0.08]

        # 4 Main fingers
        bases = [(-0.20, -0.65), (0.00, -0.70), (0.20, -0.65), (0.38, -0.58)]
        finger_states = [index_state, middle_state, ring_state, pinky_state]
        finger_ids = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]

        for idx, (mcp_id, pip_id, dip_id, tip_id) in enumerate(finger_ids):
            bx, by = bases[idx]
            state = finger_states[idx]
            lm[mcp_id] = [bx, by, 0.0]

            spread_offset = (idx - 1.5) * spread
            if crossed and idx == 0:
                spread_offset += 0.12  # Cross index over middle
            elif crossed and idx == 1:
                spread_offset -= 0.12

            if hook_index and idx == 0:
                lm[pip_id] = [bx, by - 0.22, 0.0]
                lm[dip_id] = [bx + 0.10, by - 0.32, 0.08]
                lm[tip_id] = [bx + 0.05, by - 0.20, 0.12]
            elif state == "extended":
                lm[pip_id] = [bx + spread_offset * 0.3, by - 0.25, 0.0]
                lm[dip_id] = [bx + spread_offset * 0.6, by - 0.48, 0.0]
                lm[tip_id] = [bx + spread_offset * 1.0, by - 0.70, 0.0]
            elif state == "bent":
                lm[pip_id] = [bx, by - 0.20, 0.05]
                lm[dip_id] = [bx, by - 0.22, 0.15]
                lm[tip_id] = [bx, by - 0.10, 0.18]
            elif state == "forward":
                lm[pip_id] = [bx + 0.20, by, 0.0]
                lm[dip_id] = [bx + 0.40, by, 0.0]
                lm[tip_id] = [bx + 0.60, by, 0.0]
            elif state == "down":
                lm[pip_id] = [bx, by + 0.20, 0.0]
                lm[dip_id] = [bx, by + 0.40, 0.0]
                lm[tip_id] = [bx, by + 0.60, 0.0]
            else:  # folded
                lm[pip_id] = [bx, by - 0.14, 0.08]
                lm[dip_id] = [bx, by - 0.04, 0.16]
                lm[tip_id] = [bx, by + 0.06, 0.12]

        return lm

    def render_sign_image(self, sign_name: str, width: int = 440, height: int = 440) -> np.ndarray:
        """
        Renders a beautiful visual diagram of the hand skeleton and joints.
        Returns: BGR numpy image array.
        """
        sign_key = sign_name.upper().strip()
        if sign_key in self.cached_images:
            return self.cached_images[sign_key].copy()

        # Canvas with dark gradient background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (20, 22, 28)

        # Draw grid lines for high-tech aesthetic
        for gy in range(0, height, 40):
            cv2.line(img, (0, gy), (width, gy), (28, 32, 42), 1)
        for gx in range(0, width, 40):
            cv2.line(img, (gx, 0), (gx, height), (28, 32, 42), 1)

        # Get 3D landmarks
        lm_3d = self.generate_landmark_pose(sign_key)

        # Map 3D points (-1.0..1.0) into 2D canvas pixel coordinates
        # Center of palm positioned at (width/2, height * 0.68)
        center_x = width // 2
        center_y = int(height * 0.68)
        scale = width * 0.38

        pixels = []
        for pt in lm_3d:
            px = int(center_x + pt[0] * scale)
            py = int(center_y + pt[1] * scale)
            pixels.append((px, py))

        # Draw hand bones with color-coded fingers
        for p1_id, p2_id in HAND_CONNECTIONS:
            color = FINGER_COLORS['palm']
            if p2_id in [1, 2, 3, 4]:
                color = FINGER_COLORS['thumb']
            elif p2_id in [6, 7, 8]:
                color = FINGER_COLORS['index']
            elif p2_id in [10, 11, 12]:
                color = FINGER_COLORS['middle']
            elif p2_id in [14, 15, 16]:
                color = FINGER_COLORS['ring']
            elif p2_id in [18, 19, 20]:
                color = FINGER_COLORS['pinky']

            cv2.line(img, pixels[p1_id], pixels[p2_id], color, 4, cv2.LINE_AA)

        # Draw joint nodes
        for idx, (px, py) in enumerate(pixels):
            # Larger node for wrist and tips
            radius = 6 if idx in [0, 4, 8, 12, 16, 20] else 4
            cv2.circle(img, (px, py), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), radius, (0, 240, 255) if idx in [4, 8, 12, 16, 20] else (255, 255, 255), -1, cv2.LINE_AA)

        # Header banner with sign title
        cv2.rectangle(img, (15, 15), (width - 15, 75), (32, 36, 48), -1)
        cv2.rectangle(img, (15, 15), (width - 15, 75), (60, 68, 88), 1)

        is_letter = len(sign_key) == 1 and sign_key.isalpha()
        category_tag = "ASL ALPHABET" if is_letter else "SIGN PHRASE"
        cv2.putText(img, category_tag, (28, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f'"{sign_key}"', (28, 65), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Footer banner with physical description
        desc = self.dictionary_descriptions.get(sign_key, "American Sign Language standard representation")
        cv2.rectangle(img, (15, height - 55), (width - 15, height - 15), (32, 36, 48), -1)
        cv2.putText(img, desc, (24, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 210), 1, cv2.LINE_AA)

        # Cache image
        self.cached_images[sign_key] = img
        return img

    def translate_text(self, text: str) -> list:
        """
        Parses text sentence into a sequence of sign tokens.
        Checks for whole-phrase matches first, then falls back to fingerspelling.
        Returns: list of dicts [{'token': str, 'type': 'word'|'letter'|'space', 'description': str}]
        """
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', text.upper()).strip()
        if not cleaned:
            return []

        tokens = []
        words = cleaned.split()

        # Checkable multi-word & single-word signs
        whole_word_signs = {
            'HELLO': 'HELLO',
            'HI': 'HELLO',
            'HEY': 'HELLO',
            'THANK': 'THANK YOU',
            'THANKS': 'THANK YOU',
            'YES': 'YES',
            'NO': 'NO',
            'HELP': 'HELP',
            'WATER': 'WATER',
            'DRINK': 'WATER',
            'OK': 'OK',
            'OKAY': 'OK',
            'PEACE': 'PEACE',
            'CALL': 'CALL',
            'PHONE': 'CALL',
            'PLEASE': 'PLEASE',
            'GOOD': 'GOOD',
            'BAD': 'BAD'
        }

        i = 0
        while i < len(words):
            word = words[i]

            # Check 2-word phrase: e.g. "THANK YOU", "I LOVE YOU"
            if i + 2 < len(words) and f"{words[i]} {words[i+1]} {words[i+2]}" == "I LOVE YOU":
                tokens.append({
                    'token': 'I LOVE YOU',
                    'type': 'phrase',
                    'description': self.dictionary_descriptions.get('I LOVE YOU', '')
                })
                i += 3
            elif i + 1 < len(words) and f"{words[i]} {words[i+1]}" == "THANK YOU":
                tokens.append({
                    'token': 'THANK YOU',
                    'type': 'phrase',
                    'description': self.dictionary_descriptions.get('THANK YOU', '')
                })
                i += 2
            elif word in whole_word_signs:
                sign_symbol = whole_word_signs[word]
                tokens.append({
                    'token': sign_symbol,
                    'type': 'word',
                    'description': self.dictionary_descriptions.get(sign_symbol, '')
                })
                i += 1
            else:
                # Fingerspell letter-by-letter
                for ch in word:
                    if ch.isalpha():
                        tokens.append({
                            'token': ch,
                            'type': 'letter',
                            'description': self.dictionary_descriptions.get(ch, f'ASL Manual Letter {ch}')
                        })
                i += 1

            # Append Space token between words if more words remain
            if i < len(words):
                tokens.append({
                    'token': 'SPACE',
                    'type': 'space',
                    'description': 'Pause between words'
                })

        return tokens

    def speech_to_text(self, timeout: int = 5, phrase_time_limit: int = 8) -> str:
        """
        Listens from default system microphone and transcribes speech to text.
        """
        try:
            import speech_recognition as sr
        except ImportError:
            return "[Error] SpeechRecognition library is not installed."

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True

        try:
            with sr.Microphone() as source:
                print("[STT] Adjusting for ambient noise... Please wait.")
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
                print("[STT] Listening for speech... Speak now!")
                audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                print("[STT] Processing audio transcription...")
                text = recognizer.recognize_google(audio_data)
                print(f"[STT] Transcribed: \"{text}\"")
                return text
        except sr.WaitTimeoutError:
            return "[Warning] No speech detected (timed out)."
        except sr.UnknownValueError:
            return "[Warning] Could not understand the audio speech."
        except sr.RequestError as e:
            return f"[Error] Google Speech Recognition service error: {e}"
        except Exception as e:
            return f"[Error] Microphone access error: {e}"

    def pregenerate_all_signs(self):
        """Pre-renders and saves PNG cards for all standard vocabulary in assets/signs/."""
        all_signs = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
            "HELLO", "THANK YOU", "YES", "NO", "HELP", "WATER",
            "I LOVE YOU", "OK", "PEACE", "CALL", "PLEASE", "GOOD", "BAD", "SPACE"
        ]
        print(f"[T2SL] Pre-rendering {len(all_signs)} sign asset diagrams...")
        for sign in all_signs:
            img = self.render_sign_image(sign)
            filepath = os.path.join(self.assets_dir, f"{sign}.png")
            cv2.imwrite(filepath, img)
        print(f"[T2SL] Saved all sign diagrams to {self.assets_dir}!")


if __name__ == "__main__":
    engine = TextToSignEngine()
    engine.pregenerate_all_signs()
    sample = "Hello water please"
    trans = engine.translate_text(sample)
    print(f"\nTranslation sequence for '{sample}':")
    for t in trans:
        print(f" -> [{t['type'].upper()}] {t['token']}: {t['description']}")
