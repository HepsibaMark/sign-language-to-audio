"""
Text-to-Speech (TTS) Engine Module
Provides an asynchronous, non-blocking audio synthesis queue so audio playback
never interrupts or freezes the real-time video stream.
"""

import queue
import threading
import time
import pyttsx3
try:
    import pythoncom
except ImportError:
    pythoncom = None


class TTSEngine:
    """Thread-safe, non-blocking Text-to-Speech Engine."""

    def __init__(self, rate: int = 160, volume: float = 1.0, voice_index: int = 0):
        self._queue = queue.Queue()
        self._running = True
        self._is_speaking = False
        self._rate = rate
        self._volume = volume
        self._voice_index = voice_index
        self._last_spoken_text = ""

        # Fetch available voices upfront
        self.available_voices = []
        try:
            temp_engine = pyttsx3.init()
            voices = temp_engine.getProperty('voices')
            self.available_voices = [{'id': v.id, 'name': v.name} for v in voices]
            del temp_engine
        except Exception as e:
            print(f"[TTS] Warning: Failed to query system voices: {e}")

        # Start dedicated background worker
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def _run_worker(self):
        """Worker thread loop consuming speech tasks from queue."""
        if pythoncom:
            pythoncom.CoInitialize()

        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self._rate)
            engine.setProperty('volume', self._volume)
            voices = engine.getProperty('voices')
            if voices and 0 <= self._voice_index < len(voices):
                engine.setProperty('voice', voices[self._voice_index].id)
        except Exception as e:
            print(f"[TTS] Worker initialization error: {e}")

        while self._running:
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None:
                break

            action, data = task
            if action == 'SPEAK' and engine is not None:
                text = data
                self._is_speaking = True
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as err:
                    print(f"[TTS] Error during speech synthesis: {err}")
                finally:
                    self._is_speaking = False
                    self._last_spoken_text = text
            elif action == 'SET_RATE' and engine is not None:
                self._rate = data
                engine.setProperty('rate', self._rate)
            elif action == 'SET_VOLUME' and engine is not None:
                self._volume = data
                engine.setProperty('volume', self._volume)
            elif action == 'SET_VOICE' and engine is not None:
                self._voice_index = data
                voices = engine.getProperty('voices')
                if voices and 0 <= self._voice_index < len(voices):
                    engine.setProperty('voice', voices[self._voice_index].id)

            self._queue.task_done()

        if pythoncom:
            pythoncom.CoUninitialize()

    def speak(self, text: str, min_interval_seconds: float = 0.5):
        """
        Enqueue text to be spoken.
        Skips empty text or immediate duplicate spam.
        """
        cleaned = text.strip()
        if not cleaned:
            return False

        # Put speech command into queue
        self._queue.put(('SPEAK', cleaned))
        return True

    def set_rate(self, rate: int):
        """Adjust speech rate (speed). Typical range: 100 - 250 wpm."""
        self._queue.put(('SET_RATE', rate))

    def set_volume(self, volume: float):
        """Adjust volume (0.0 to 1.0)."""
        self._queue.put(('SET_VOLUME', max(0.0, min(1.0, volume))))

    def set_voice(self, voice_index: int):
        """Select voice by index."""
        self._queue.put(('SET_VOICE', voice_index))

    @property
    def is_speaking(self) -> bool:
        """Returns True if the engine is actively speaking audio."""
        return self._is_speaking

    @property
    def last_spoken(self) -> str:
        return self._last_spoken_text

    def stop(self):
        """Gracefully stop TTS worker."""
        self._running = False
        self._queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
