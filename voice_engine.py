"""
Voice Engine - High Quality TTS & Voice Cloning with Emotions
Uses Coqui XTTS v2 for state-of-the-art open source voice synthesis

UPGRADES v2:
- Fixed temperature param being silently ignored
- Fixed makedirs crash on bare filenames
- Long-form audio: auto-splits text into chunks, generates each, stitches together
  → supports 20-30+ minute audio generation
- Noise reduction preprocessing on uploaded voice samples
- Named voice profile saving/loading
- Batch export to MP3 as well as WAV
- Sentence-aware chunking (no word cutoffs)
- Cross-fade stitching for smooth joins between chunks
"""

import os
import re
import time
import torch
import shutil
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path
from TTS.api import TTS


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _safe_makedirs(path: str):
    """Create parent directories safely even if path has no directory component."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _split_text_into_chunks(text: str, max_words: int = 200) -> list[str]:
    """
    Split long text into sentence-aware chunks of at most max_words words.
    Never cuts mid-sentence. Handles paragraphs, ellipses, abbreviations.
    Supports 20-30+ minute audio generation when max_words=200.
    """
    # Normalize whitespace
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text.strip())

    # Split into sentences (keeps delimiter attached)
    sentence_endings = re.compile(r'(?<=[.!?…])\s+')
    raw_sentences = sentence_endings.split(text)

    # Also split on double newlines (paragraph breaks)
    sentences = []
    for s in raw_sentences:
        parts = s.split('\n\n')
        sentences.extend(p.strip() for p in parts if p.strip())

    chunks = []
    current_words = []
    current_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        # If a single sentence is larger than max_words, split it by commas
        if word_count > max_words:
            sub_parts = re.split(r',\s*', sentence)
            for part in sub_parts:
                pc = len(part.split())
                if current_count + pc > max_words and current_words:
                    chunks.append(' '.join(current_words))
                    current_words = [part]
                    current_count = pc
                else:
                    current_words.append(part)
                    current_count += pc
        elif current_count + word_count > max_words:
            if current_words:
                chunks.append(' '.join(current_words))
            current_words = [sentence]
            current_count = word_count
        else:
            current_words.append(sentence)
            current_count += word_count

    if current_words:
        chunks.append(' '.join(current_words))

    return [c for c in chunks if c.strip()]


def _crossfade_join(arrays: list[np.ndarray], sample_rate: int, fade_ms: int = 80) -> np.ndarray:
    """
    Join audio arrays with a short linear crossfade to avoid clicks/pops.
    fade_ms: crossfade window in milliseconds
    """
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples == 0 or len(arrays) == 1:
        return np.concatenate(arrays)

    result = arrays[0]
    for nxt in arrays[1:]:
        if len(result) < fade_samples or len(nxt) < fade_samples:
            result = np.concatenate([result, nxt])
            continue
        # Build fade curves
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in  = np.linspace(0.0, 1.0, fade_samples)
        # Overlap-add
        overlap = result[-fade_samples:] * fade_out + nxt[:fade_samples] * fade_in
        result = np.concatenate([result[:-fade_samples], overlap, nxt[fade_samples:]])

    return result


def _try_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply noise reduction if noisereduce is installed, else return as-is."""
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=sr, stationary=False)
    except ImportError:
        return audio


# ──────────────────────────────────────────────────────────
# VoiceEngine
# ──────────────────────────────────────────────────────────

class VoiceEngine:
    # Emotion presets ─ temperature controls expressiveness, speed controls pace
    EMOTION_PRESETS = {
        "neutral":  {"temperature": 0.70, "speed": 1.00},
        "happy":    {"temperature": 0.80, "speed": 1.05},
        "sad":      {"temperature": 0.50, "speed": 0.85},
        "angry":    {"temperature": 0.90, "speed": 1.10},
        "excited":  {"temperature": 0.95, "speed": 1.15},
        "whisper":  {"temperature": 0.40, "speed": 0.70},
    }

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        gpu: bool = False,
        profiles_dir: str = "voice_profiles",
    ):
        """
        Initialize the voice engine.

        Args:
            model_name : XTTS v2 model
            gpu        : Use GPU if available (strongly recommended for long audio)
            profiles_dir: Folder where named voice profiles are stored
        """
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = 24000

        print(f"[VoiceEngine] Loading model on: {self.device}")
        print("[VoiceEngine] First run downloads ~1.5 GB model files …")
        self.tts = TTS(model_name).to(self.device)
        print("[VoiceEngine] Model ready.")

    # ── Public API ──────────────────────────────────────────

    def generate_speech(
        self,
        text: str,
        speaker_wav: str | None = None,
        language: str = "en",
        speed: float = 1.0,
        temperature: float = 0.7,
        output_path: str = "output/generated.wav",
        fmt: str = "wav",
        progress_callback=None,
    ) -> str:
        """
        Generate speech for ANY length of text.

        Long texts are automatically chunked into ≤200-word segments,
        synthesised individually, then crossfade-stitched into one file.
        This allows 20-30+ minute audio generation with no quality loss.

        Args:
            text            : Text to synthesise (no length limit)
            speaker_wav     : Reference voice sample path (enables cloning)
            language        : BCP-47 language code
            speed           : 0.5–2.0 (applies to every chunk)
            temperature     : 0.0–1.0 — expressiveness (now actually used ✓)
            output_path     : Destination file (.wav or .mp3)
            fmt             : "wav" or "mp3"
            progress_callback: Optional fn(current, total, chunk_text) for UI updates
        Returns:
            Absolute path of saved audio file
        """
        _safe_makedirs(output_path)

        chunks = _split_text_into_chunks(text, max_words=200)
        total  = len(chunks)
        print(f"[VoiceEngine] {total} chunk(s) to synthesise …")

        chunk_arrays: list[np.ndarray] = []
        tmp_dir = tempfile.mkdtemp(prefix="vc_chunks_")

        try:
            for i, chunk in enumerate(chunks, 1):
                if progress_callback:
                    progress_callback(i, total, chunk[:60])
                print(f"[VoiceEngine] Chunk {i}/{total}: "{chunk[:60]}…"")

                chunk_path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")

                if speaker_wav and os.path.exists(speaker_wav):
                    self.tts.tts_to_file(
                        text=chunk,
                        speaker_wav=speaker_wav,
                        language=language,
                        file_path=chunk_path,
                        speed=speed,
                        temperature=temperature,   # ← BUG FIX: now passed correctly
                    )
                else:
                    self.tts.tts_to_file(
                        text=chunk,
                        file_path=chunk_path,
                        speaker="Ana Florence",
                        language=language,
                        speed=speed,
                        temperature=temperature,   # ← BUG FIX
                    )

                audio, sr = sf.read(chunk_path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)   # stereo → mono
                chunk_arrays.append(audio)

            # Stitch with crossfade
            print("[VoiceEngine] Stitching chunks …")
            full_audio = _crossfade_join(chunk_arrays, self.sample_rate)

            # Write final file
            final_wav = output_path if output_path.endswith(".wav") else output_path + ".wav"
            sf.write(final_wav, full_audio, self.sample_rate)

            if fmt == "mp3":
                mp3_path = re.sub(r'\.wav$', '.mp3', final_wav)
                self._wav_to_mp3(final_wav, mp3_path)
                os.remove(final_wav)
                output_path = mp3_path
            else:
                output_path = final_wav

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        duration_s = len(full_audio) / self.sample_rate
        mins, secs = divmod(int(duration_s), 60)
        print(f"[VoiceEngine] Done → {output_path}  ({mins}m {secs}s)")
        return output_path

    def clone_with_emotion(
        self,
        text: str,
        speaker_wav: str,
        emotion_hint: str = "neutral",
        language: str = "en",
        output_path: str = "output/emotional.wav",
        fmt: str = "wav",
        progress_callback=None,
    ) -> str:
        """Clone voice with emotional colouring. Supports unlimited text length."""
        settings     = self.EMOTION_PRESETS.get(emotion_hint, self.EMOTION_PRESETS["neutral"])
        enhanced_txt = self._enhance_text_emotion(text, emotion_hint)

        return self.generate_speech(
            text=enhanced_txt,
            speaker_wav=speaker_wav,
            language=language,
            speed=settings["speed"],
            temperature=settings["temperature"],
            output_path=output_path,
            fmt=fmt,
            progress_callback=progress_callback,
        )

    # ── Voice Profiles ──────────────────────────────────────

    def save_voice_profile(self, name: str, source_wav: str) -> str:
        """
        Save a voice sample under a named profile for later reuse.
        Audio is denoised before saving for best cloning quality.
        Returns path of the saved profile.
        """
        name = re.sub(r'[^\w\-]', '_', name)
        dest = self.profiles_dir / f"{name}.wav"
        audio, sr = sf.read(source_wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _try_denoise(audio, sr)
        sf.write(str(dest), audio, sr)
        print(f"[VoiceEngine] Profile saved: {dest}")
        return str(dest)

    def list_voice_profiles(self) -> list[str]:
        """Return names of saved voice profiles (without extension)."""
        return [p.stem for p in self.profiles_dir.glob("*.wav")]

    def get_profile_path(self, name: str) -> str | None:
        """Return full path of a named profile, or None if not found."""
        path = self.profiles_dir / f"{name}.wav"
        return str(path) if path.exists() else None

    def delete_voice_profile(self, name: str) -> bool:
        """Delete a saved voice profile. Returns True if deleted."""
        path = self.profiles_dir / f"{name}.wav"
        if path.exists():
            path.unlink()
            return True
        return False

    # ── Utilities ───────────────────────────────────────────

    def preprocess_voice_sample(self, wav_path: str, output_path: str | None = None) -> str:
        """
        Denoise and normalise a voice sample for better cloning.
        Saves to output_path (or overwrites source if None).
        """
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _try_denoise(audio, sr)
        # Peak-normalise to -1 dBFS
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.891
        dest = output_path or wav_path
        _safe_makedirs(dest)
        sf.write(dest, audio, sr)
        return dest

    def estimate_audio_duration(self, text: str, speed: float = 1.0) -> float:
        """
        Estimate output audio duration in seconds based on word count.
        Average English speech ≈ 140 words/min at speed=1.0.
        """
        words = len(text.split())
        base_duration = (words / 140) * 60
        return base_duration / speed

    def list_languages(self) -> list[str]:
        return ["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh","ja","hu","ko"]

    @staticmethod
    def _wav_to_mp3(wav_path: str, mp3_path: str):
        """Convert WAV to MP3 using pydub (requires ffmpeg)."""
        try:
            from pydub import AudioSegment
            AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate="192k")
        except Exception as e:
            print(f"[VoiceEngine] MP3 conversion failed ({e}). Keeping WAV.")
            shutil.copy(wav_path, mp3_path.replace(".mp3", ".wav"))

    @staticmethod
    def _enhance_text_emotion(text: str, emotion: str) -> str:
        """Add subtle punctuation cues if none match the target emotion."""
        if emotion == "excited" and "!" not in text:
            text = text.strip() + "!"
        elif emotion == "sad" and "..." not in text:
            text = text.strip() + "..."
        elif emotion == "whisper":
            text = f"(whispering) {text}"
        return text


# ── Quick smoke-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = VoiceEngine(gpu=False)

    # Short test
    engine.generate_speech(
        text="Hello! This is a test of the upgraded voice generation engine.",
        output_path="output/test_default.wav",
    )

    # Long-form test (~1 min)
    long_text = ("This is a long-form audio generation test. " * 80).strip()
    engine.generate_speech(
        text=long_text,
        output_path="output/test_long.wav",
    )
    print("All tests passed.")
