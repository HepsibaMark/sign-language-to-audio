"""
Test script for Text-to-Speech (TTS) Engine
"""

import time
from src.tts_engine import TTSEngine


def main():
    print("Initializing Text-to-Speech Engine...")
    tts = TTSEngine(rate=160, volume=1.0)

    print("\nAvailable Voices:")
    for idx, v in enumerate(tts.available_voices):
        print(f" [{idx}] {v['name']}")

    test_phrases = [
        "Welcome to the Sign Language to Audio assistive system.",
        "Hello, my name is Alex.",
        "I need water, please.",
        "Thank you for bridging communication barriers."
    ]

    print("\nSynthesizing test speech phrases...")
    for phrase in test_phrases:
        print(f" -> Speaking: \"{phrase}\"")
        tts.speak(phrase)
        time.sleep(3.0)

    tts.stop()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
