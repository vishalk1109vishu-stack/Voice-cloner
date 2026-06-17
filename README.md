# 🎙️ Voice Cloning Studio v2

A complete Python project for high-quality AI voice generation and voice cloning.
Powered by **Coqui XTTS v2** — the best open-source voice synthesis engine.

---

## ✨ What's new in v2

| Change | Details |
|--------|---------|
| **Unlimited text length** | Auto-chunks into ≤200-word segments, synthesises each, then crossfade-stitches into one seamless file — 20–30 min audio with a single click |
| **Temperature bug fixed** | Emotion expressiveness now actually works (was silently ignored in v1) |
| **Live duration estimate** | See "~14m 30s (2100 words, 11 chunks)" before you generate |
| **Named voice profiles** | Save voice samples under a name — reuse across sessions without re-uploading |
| **Auto noise reduction** | Uploaded samples are denoised + normalised before cloning |
| **MP3 export** | Generate MP3 directly (requires ffmpeg) as well as WAV |
| **Custom filenames** | Name your output files instead of generic timestamps |
| **Generation history** | Last 50 jobs logged with duration, language, mode, and filename |
| **Gradio 4 fix** | `source=` → `sources=["upload"]` deprecation fixed |
| **`makedirs` crash fix** | No longer crashes on bare filenames with no directory component |
| **Speed applies to cloning** | Speed slider now works in cloning mode too (was ignored in v1) |
| **Cleaner requirements.txt** | Removed `pathlib` (it's stdlib, not a pip package) |

---

## 🚀 Quick Start

### 1. Install

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

**Optional extras:**
```bash
pip install noisereduce   # better voice cloning quality (auto-denoising)
pip install pydub         # MP3 export (also needs ffmpeg on your system)
```

**GPU (10× faster):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# then set gpu=True in app.py line:  engine = VoiceEngine(gpu=True, ...)
```

### 2. Run

```bash
python app.py
```

Opens a browser UI. First launch downloads ~1.5 GB of model files (one-time).

---

## 📁 Project Structure

```
voice_cloning_project/
├── voice_engine.py          # Core engine — TTS, cloning, chunking, profiles
├── app.py                   # Gradio web UI
├── requirements.txt         # Python dependencies
├── generation_history.json  # Auto-created: last 50 generation records
├── samples/                 # Temporary uploaded audio
├── output/                  # All generated audio files
└── voice_profiles/          # Saved named voice profiles (auto-created)
```

---

## 🧩 API Usage

```python
from voice_engine import VoiceEngine

engine = VoiceEngine(gpu=False)   # gpu=True if you have NVIDIA CUDA

# ── Basic TTS ──────────────────────────────────────────────────────────────
engine.generate_speech(
    text="Hello, this is AI speaking!",
    language="en",
    output_path="output/hello.wav",
)

# ── Long-form (20–30 min) ─────────────────────────────────────────────────
with open("my_long_script.txt") as f:
    script = f.read()

engine.generate_speech(
    text=script,
    language="en",
    speed=1.0,
    temperature=0.7,
    output_path="output/full_podcast.wav",
    progress_callback=lambda cur, tot, snippet:
        print(f"Chunk {cur}/{tot}: {snippet[:50]}"),
)

# ── Voice Cloning with Emotion ────────────────────────────────────────────
engine.clone_with_emotion(
    text="I can sound just like you, with emotions!",
    speaker_wav="samples/my_voice.wav",
    emotion_hint="excited",
    output_path="output/cloned_excited.wav",
)

# ── Named Voice Profiles ──────────────────────────────────────────────────
engine.save_voice_profile("alice", "samples/alice.wav")   # saves + denoises
path = engine.get_profile_path("alice")
engine.generate_speech("Hello from Alice.", speaker_wav=path, output_path="output/alice_says_hello.wav")

print(engine.list_voice_profiles())   # ['alice', ...]
engine.delete_voice_profile("alice")

# ── Estimate duration before generating ───────────────────────────────────
secs = engine.estimate_audio_duration(text=script, speed=1.0)
print(f"Expected: {secs/60:.1f} minutes")
```

---

## 🎭 Emotion Guide

| Emotion  | Temperature | Speed | Best For                   |
|----------|-------------|-------|----------------------------|
| neutral  | 0.70        | 1.00  | Narration, articles        |
| happy    | 0.80        | 1.05  | Cheerful announcements     |
| sad      | 0.50        | 0.85  | Emotional stories          |
| angry    | 0.90        | 1.10  | Intense scenes             |
| excited  | 0.95        | 1.15  | Energetic content          |
| whisper  | 0.40        | 0.70  | ASMR, secretive dialogue   |

---

## 🌍 Supported Languages

English · Spanish · French · German · Italian · Portuguese · Polish · Turkish ·
Russian · Dutch · Czech · Arabic · Chinese · Japanese · Hungarian · Korean

---

## 💻 Hardware Requirements

| Setup              | Synthesis speed         | Recommendation        |
|--------------------|-------------------------|-----------------------|
| CPU                | ~1–2 min per min audio  | Testing / short clips |
| GPU (4 GB+ VRAM)   | ~5–10 sec per min audio | Production / long audio |

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Use CPU mode or reduce `max_words` per chunk in `_split_text_into_chunks` |
| `Model download failed` | Check internet connection; retry |
| `Audio sounds robotic` | Use a longer, cleaner voice sample; enable `noisereduce` |
| `MP3 export fails` | Install `pydub` and `ffmpeg` (`brew install ffmpeg` / `apt install ffmpeg`) |
| Slow on CPU | Expected — ~1–2 min per minute of output. Use GPU for large jobs. |

---

## ⚠️ Legal & Ethical Use

- Only clone voices you have explicit permission to reproduce
- This project uses Coqui TTS (CPML licence)
- For commercial production, consider ElevenLabs or similar licensed APIs
