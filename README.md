# 🎙️ Voice Cloning Studio

A complete Python project for high-quality AI voice generation and voice cloning with emotional control. Built on Coqui XTTS v2 — the best open-source voice synthesis engine.

## ✨ Features

- **Text-to-Speech**: Generate natural, human-like speech in 16 languages
- **Voice Cloning**: Clone any voice from just 6 seconds of audio
- **Emotion Control**: Adjust tone — neutral, happy, sad, angry, excited, whisper
- **Web UI**: Easy-to-use Gradio interface (no coding needed)
- **API Mode**: Programmatic access via Python class

## 🚀 Quick Start

### 1. Install Requirements

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: First run downloads ~1.5GB of model files automatically.

### 2. Run the Web Interface

```bash
python app.py
```

This opens a web UI in your browser. You can:
- Type text and generate speech instantly
- Upload a voice sample to clone it
- Select emotion styles when cloning

### 3. Use the API in Your Code

```python
from voice_engine import VoiceEngine

# Initialize
engine = VoiceEngine(gpu=False)  # Set gpu=True if you have NVIDIA GPU

# Basic TTS
engine.generate_speech(
    text="Hello, this is AI speaking!",
    language="en",
    output_path="output/hello.wav"
)

# Voice Cloning with Emotion
engine.clone_with_emotion(
    text="I can sound just like you, with emotions!",
    speaker_wav="samples/my_voice.wav",
    emotion_hint="excited",
    output_path="output/cloned_excited.wav"
)
```

## 📁 Project Structure

```
voice_cloning_project/
├── voice_engine.py      # Core engine (TTS + cloning + emotions)
├── app.py               # Gradio web interface
├── requirements.txt     # Python dependencies
├── samples/             # Put voice samples here for cloning
└── output/              # Generated audio files go here
```

## 🎭 Emotion Guide

| Emotion  | Effect                                | Best For                  |
|----------|---------------------------------------|---------------------------|
| neutral  | Natural, balanced tone                | General narration         |
| happy    | Slightly faster, warmer tone          | Cheerful content          |
| sad      | Slower, softer delivery               | Emotional stories         |
| angry    | Faster, more forceful                 | Intense scenes            |
| excited  | Fast, energetic with exclamation      | Enthusiastic announcements|
| whisper  | Very slow, quiet delivery             | ASMR, secretive dialogue  |

**Pro tip**: The reference audio's natural emotion has the biggest impact. For best results, upload a sample that matches the emotion you want.

## 🌍 Supported Languages

English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean

## 💻 Hardware Requirements

| Setup  | Speed    | Quality | Recommendation        |
|--------|----------|---------|----------------------|
| CPU    | ~1-2 min | Good    | Works for testing    |
| GPU    | ~5-10 sec| Best    | NVIDIA with 4GB+ VRAM|

To enable GPU:
1. Install NVIDIA drivers
2. Install CUDA toolkit
3. Install PyTorch with CUDA: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118`
4. Set `gpu=True` in VoiceEngine()

## 📝 Tips for Best Voice Cloning

1. **Audio Quality**: Use clean, noise-free recordings
2. **Length**: 6-30 seconds is the sweet spot
3. **Single Speaker**: No background voices or music
4. **Consistency**: One continuous sentence works better than chopped clips
5. **Format**: WAV is best, but MP3 works too

## ⚠️ Important Notes

- **First run is slow**: The model (~1.5GB) downloads automatically
- **Legal**: Only clone voices you have permission to use
- **Quality**: Open-source models are good but not quite ElevenLabs-level. For commercial production, consider their API.

## 🔧 Troubleshooting

**"CUDA out of memory"** → Reduce batch size or use CPU mode
**"Model download failed"** → Check internet connection, retry
**"Audio sounds robotic"** → Use longer, cleaner voice samples
**"Slow on CPU"** → This is normal. First generation takes 2-3 minutes.

## 📜 License

This project uses Coqui TTS (CPML license). Voice cloning should only be used ethically and legally.
