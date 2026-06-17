"""
Voice Engine - High Quality TTS & Voice Cloning with Emotions
Uses Coqui XTTS v2 for state-of-the-art open source voice synthesis
"""
import os
import torch
import soundfile as sf
from TTS.api import TTS
from pathlib import Path

class VoiceEngine:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False):
        """
        Initialize the voice engine.

        Args:
            model_name: XTTS v2 model (multilingual, supports voice cloning)
            gpu: Use GPU if available (much faster, recommended)
        """
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        print(f"Loading model on: {self.device}")
        print("First run will download ~1.5GB model files...")

        self.tts = TTS(model_name).to(self.device)
        self.sample_rate = 24000

    def generate_speech(self, text, speaker_wav=None, language="en", 
                       speed=1.0, temperature=0.7, output_path="output/generated.wav"):
        """
        Generate high-quality speech. Clone voice if speaker_wav provided.

        Args:
            text: Text to speak
            speaker_wav: Path to reference voice sample (for cloning). If None, uses default voice.
            language: Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko)
            speed: Speaking speed (0.5 to 2.0). 1.0 is normal.
            temperature: Controls randomness/expressiveness (0.0 to 1.0). Higher = more emotional variation.
            output_path: Where to save the audio file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if speaker_wav and os.path.exists(speaker_wav):
            # Voice cloning mode
            print(f"Cloning voice from: {speaker_wav}")
            self.tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path,
                speed=speed
            )
        else:
            # Default voice mode (no cloning)
            print("Using default voice (no cloning)")
            # For XTTS without speaker reference, we use a built-in speaker
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker="Ana Florence",  # Built-in speaker
                language=language
            )

        print(f"Audio saved to: {output_path}")
        return output_path

    def clone_with_emotion(self, text, speaker_wav, emotion_hint="neutral", 
                          language="en", output_path="output/emotional.wav"):
        """
        Clone voice with emotional coloring.

        Emotion is controlled by:
        1. The reference audio's natural emotion (most important)
        2. Text punctuation and formatting (exclamation marks, caps, etc.)
        3. Temperature settings

        Args:
            text: Text to speak (use punctuation for emotion!)
            speaker_wav: Reference voice sample
            emotion_hint: "neutral", "happy", "sad", "angry", "excited" (affects temperature)
            language: Language code
            output_path: Output file path
        """
        # Emotion presets via temperature and speed
        emotion_settings = {
            "neutral": {"temperature": 0.7, "speed": 1.0},
            "happy": {"temperature": 0.8, "speed": 1.05},
            "sad": {"temperature": 0.5, "speed": 0.85},
            "angry": {"temperature": 0.9, "speed": 1.1},
            "excited": {"temperature": 0.95, "speed": 1.15},
            "whisper": {"temperature": 0.4, "speed": 0.7},
        }

        settings = emotion_settings.get(emotion_hint, emotion_settings["neutral"])

        # Enhance text with emotional markers if not present
        enhanced_text = self._enhance_text_emotion(text, emotion_hint)

        return self.generate_speech(
            text=enhanced_text,
            speaker_wav=speaker_wav,
            language=language,
            speed=settings["speed"],
            temperature=settings["temperature"],
            output_path=output_path
        )

    def _enhance_text_emotion(self, text, emotion):
        """Add subtle emotional cues to text if none exist."""
        if emotion == "excited" and not any(c in text for c in "!"):
            text = text.strip() + "!"
        elif emotion == "sad" and not any(c in text for c in "..."):
            text = text.strip() + "..."
        elif emotion == "whisper":
            text = f"(whispering) {text}"
        return text

    def list_languages(self):
        """Return supported languages."""
        return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"]

if __name__ == "__main__":
    # Quick test
    engine = VoiceEngine(gpu=False)

    # Test 1: Default voice
    engine.generate_speech(
        text="Hello! This is a test of high quality voice generation.",
        output_path="output/test_default.wav"
    )

    # Test 2: Voice cloning (requires sample in samples/ folder)
    if os.path.exists("samples/my_voice.wav"):
        engine.clone_with_emotion(
            text="This is my voice being cloned by artificial intelligence!",
            speaker_wav="samples/my_voice.wav",
            emotion_hint="excited",
            output_path="output/test_cloned.wav"
        )
    else:
        print("\nPlace a voice sample in samples/my_voice.wav to test cloning")
