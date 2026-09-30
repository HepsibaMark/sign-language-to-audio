"""
Sentence Builder & Gesture Stabilizer Module
Smooths raw frame predictions, prevents jitter with hold-to-confirm debouncing,
and manages the dynamic sentence construction buffer.
"""

import time


class SentenceBuilder:
    """Manages sentence formation, gesture stabilization, and auto-spacing."""

    def __init__(self,
                 hold_time_seconds: float = 0.8,
                 cooldown_seconds: float = 0.6,
                 min_confidence: float = 0.75):
        self.hold_time_required = hold_time_seconds
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence

        # Tracking state
        self.current_candidate = None
        self.candidate_start_time = 0.0
        self.last_confirmed_time = 0.0
        self.last_confirmed_label = None

        # Sentence data
        self.current_sentence = ""
        self.sentence_history = []

    def update(self, detected_label: str, confidence: float) -> tuple:
        """
        Updates stabilizer with the latest frame's prediction.
        Returns:
            (confirmed_item: str or None, progress: float, is_cooldown: bool)
        """
        now = time.time()

        # Ignore low confidence or unknown
        if confidence < self.min_confidence or detected_label in ["UNKNOWN", ""]:
            self.current_candidate = None
            self.candidate_start_time = 0.0
            return None, 0.0, False

        # In cooldown period after registering a sign
        if (now - self.last_confirmed_time) < self.cooldown_seconds:
            # If the sign is the exact same as just confirmed, keep waiting
            if detected_label == self.last_confirmed_label:
                return None, 0.0, True

        # Candidate check
        if detected_label == self.current_candidate:
            elapsed = now - self.candidate_start_time
            progress = min(1.0, elapsed / self.hold_time_required)

            if progress >= 1.0:
                # Registered!
                confirmed = self._handle_confirmed(detected_label)
                self.last_confirmed_time = now
                self.last_confirmed_label = detected_label
                self.current_candidate = None
                self.candidate_start_time = 0.0
                return confirmed, 1.0, False

            return None, progress, False
        else:
            # New gesture started
            self.current_candidate = detected_label
            self.candidate_start_time = now
            return None, 0.0, False

    def _handle_confirmed(self, label: str) -> str:
        """Maps recognized gesture label into sentence buffer."""
        clean_text = label

        # Map composite labels into clean words
        if "I LOVE YOU" in label:
            clean_text = "I love you"
        elif "HELLO" in label:
            clean_text = "Hello"
        elif "HELP" in label or "GOOD" in label:
            clean_text = "Help"
        elif "BAD" in label or "NO" in label:
            clean_text = "No"
        elif "THANK YOU" in label:
            clean_text = "Thank you"
        elif "WATER" in label:
            clean_text = "Water"
        elif "PEACE" in label:
            clean_text = "Peace"
        elif "CALL" in label:
            clean_text = "Call"
        elif "YES" in label:
            clean_text = "Yes"
        elif "OK" in label:
            clean_text = "OK"

        # Check if single letter or whole word/phrase
        if len(clean_text) == 1 and clean_text.isalpha():
            self.current_sentence += clean_text
        else:
            if self.current_sentence and not self.current_sentence.endswith(" "):
                self.current_sentence += " " + clean_text + " "
            else:
                self.current_sentence += clean_text + " "

        return clean_text

    def add_space(self):
        """Manually append space."""
        if self.current_sentence and not self.current_sentence.endswith(" "):
            self.current_sentence += " "

    def backspace(self):
        """Remove last character or word."""
        if not self.current_sentence:
            return

        self.current_sentence = self.current_sentence.rstrip()
        # If there is a space, delete last word
        if " " in self.current_sentence:
            self.current_sentence = self.current_sentence.rsplit(" ", 1)[0] + " "
        else:
            self.current_sentence = self.current_sentence[:-1]

    def clear(self):
        """Clears current sentence."""
        self.current_sentence = ""

    def get_and_archive_sentence(self) -> str:
        """Returns full sentence and pushes to history."""
        sentence = self.current_sentence.strip()
        if sentence:
            self.sentence_history.append(sentence)
            self.current_sentence = ""
        return sentence

    @property
    def sentence(self) -> str:
        return self.current_sentence
