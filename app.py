"""
Voice Cloning Studio v2 — Web UI
Run: python app.py
"""

import os
import time
import json
import shutil
import hashlib
import gradio as gr
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path
from voice_engine import VoiceEngine

# ── Init ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
OUTPUT_DIR   = BASE_DIR / "output"
SAMPLES_DIR  = BASE_DIR / "samples"
HISTORY_FILE = BASE_DIR / "generation_history.json"

OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

print("Initialising Voice Engine …")
engine = VoiceEngine(gpu=False, profiles_dir=str(BASE_DIR / "voice_profiles"))

# ── Constants ────────────────────────────────────────────────────────────────
LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr",
    "Russian": "ru", "Dutch": "nl", "Czech": "cs", "Arabic": "ar",
    "Chinese": "zh", "Japanese": "ja", "Hungarian": "hu", "Korean": "ko",
}
EMOTIONS  = ["neutral", "happy", "sad", "angry", "excited", "whisper"]
FORMATS   = ["WAV", "MP3"]


# ── History helpers ──────────────────────────────────────────────────────────
def _load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            pass
    return []

def _save_history(entries):
    HISTORY_FILE.write_text(json.dumps(entries[-50:], indent=2))   # keep last 50

def _add_history(entry: dict):
    entries = _load_history()
    entries.append(entry)
    _save_history(entries)

def _history_table():
    entries = _load_history()
    if not entries:
        return []
    return [
        [
            e.get("timestamp", ""),
            e.get("mode", ""),
            e.get("language", ""),
            e.get("emotion", ""),
            e.get("duration", ""),
            e.get("output_file", ""),
        ]
        for e in reversed(entries)
    ]


# ── Profile helpers ──────────────────────────────────────────────────────────
def _profile_choices():
    return ["(none)"] + engine.list_voice_profiles()


# ── Core generation function ─────────────────────────────────────────────────
def generate_voice(
    text, speaker_audio, profile_name,
    language, emotion, speed, temperature,
    output_fmt, custom_filename,
    progress=gr.Progress(track_tqdm=False),
):
    if not text or not text.strip():
        return None, "⚠️ Please enter some text.", _history_table()

    lang_code = LANGUAGES.get(language, "en")
    fmt       = output_fmt.lower()

    # Estimate duration
    est_s    = engine.estimate_audio_duration(text, speed)
    est_min  = est_s / 60
    word_cnt = len(text.split())

    # Resolve speaker wav: uploaded audio > named profile > none
    speaker_path = None

    if speaker_audio is not None:
        tmp_wav = str(SAMPLES_DIR / "uploaded_voice.wav")
        # Gradio 4 audio is (sample_rate, numpy_array)
        if isinstance(speaker_audio, tuple):
            sr_in, arr = speaker_audio
            wavfile.write(tmp_wav, sr_in, arr)
        else:
            shutil.copy(speaker_audio, tmp_wav)
        # Denoise
        speaker_path = engine.preprocess_voice_sample(tmp_wav)

    elif profile_name and profile_name != "(none)":
        speaker_path = engine.get_profile_path(profile_name)

    # Build output filename
    if custom_filename and custom_filename.strip():
        safe_name = "".join(c for c in custom_filename.strip() if c.isalnum() or c in "-_ ")
        safe_name = safe_name.replace(" ", "_")
    else:
        ts        = time.strftime("%Y%m%d_%H%M%S")
        safe_name = f"generated_{ts}"

    output_path = str(OUTPUT_DIR / f"{safe_name}.{'mp3' if fmt == 'mp3' else 'wav'}")

    # Progress wrapper
    def on_progress(current, total, snippet):
        pct = current / total
        progress(pct, desc=f"Chunk {current}/{total}: "{snippet[:40]}…"")

    progress(0, desc="Starting generation …")

    try:
        if speaker_path:
            result = engine.clone_with_emotion(
                text=text,
                speaker_wav=speaker_path,
                emotion_hint=emotion,
                language=lang_code,
                output_path=output_path,
                fmt=fmt,
                progress_callback=on_progress,
            )
            mode = "Voice Cloning"
        else:
            result = engine.generate_speech(
                text=text,
                language=lang_code,
                speed=speed,
                temperature=temperature,
                output_path=output_path,
                fmt=fmt,
                progress_callback=on_progress,
            )
            mode = "Default Voice"

        progress(1.0, desc="Done!")

        # Actual duration from file
        import soundfile as sf
        info     = sf.info(result)
        dur_s    = info.duration
        dur_min, dur_sec = divmod(int(dur_s), 60)

        _add_history({
            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode":        mode,
            "language":    language,
            "emotion":     emotion,
            "duration":    f"{dur_min}m {dur_sec}s",
            "output_file": Path(result).name,
            "words":       word_cnt,
        })

        status = (
            f"✅ {mode} | {language} | Emotion: {emotion} | "
            f"Duration: {dur_min}m {dur_sec}s | {word_cnt} words → {Path(result).name}"
        )
        return result, status, _history_table()

    except Exception as e:
        return None, f"❌ Error: {e}", _history_table()


# ── Profile management functions ─────────────────────────────────────────────
def save_profile(speaker_audio, profile_name):
    if speaker_audio is None:
        return "⚠️ Upload a voice sample first.", _profile_choices()
    if not profile_name or not profile_name.strip():
        return "⚠️ Enter a profile name.", _profile_choices()

    tmp_wav = str(SAMPLES_DIR / "profile_tmp.wav")
    if isinstance(speaker_audio, tuple):
        sr_in, arr = speaker_audio
        wavfile.write(tmp_wav, sr_in, arr)
    else:
        shutil.copy(speaker_audio, tmp_wav)

    saved = engine.save_voice_profile(profile_name.strip(), tmp_wav)
    return f"✅ Profile '{profile_name}' saved ({Path(saved).name})", _profile_choices()


def delete_profile(profile_name):
    if not profile_name or profile_name == "(none)":
        return "⚠️ Select a profile to delete.", _profile_choices()
    ok = engine.delete_voice_profile(profile_name)
    msg = f"🗑️ Profile '{profile_name}' deleted." if ok else f"⚠️ Profile '{profile_name}' not found."
    return msg, _profile_choices()


def refresh_profiles():
    return gr.update(choices=_profile_choices(), value="(none)")


# ── Duration estimator ────────────────────────────────────────────────────────
def estimate_duration(text, speed):
    if not text:
        return "—"
    s = engine.estimate_audio_duration(text, speed)
    m, sc = divmod(int(s), 60)
    h, m  = divmod(m, 60)
    words = len(text.split())
    chunks = len([1 for _ in range(0, words, 200)])
    if h:
        return f"~{h}h {m}m {sc}s  ({words:,} words, {chunks} chunks)"
    return f"~{m}m {sc}s  ({words:,} words, {chunks} chunks)"


# ── Gradio UI ────────────────────────────────────────────────────────────────
CSS = """
/* ── Base ── */
body, .gradio-container { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

/* ── Header ── */
.studio-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border: 1px solid rgba(99,179,237,0.2);
}
.studio-header h1 { color: #e2e8f0; font-size: 2rem; font-weight: 700; margin: 0; }
.studio-header p  { color: #90cdf4; margin: 6px 0 0; font-size: 1rem; }

/* ── Panels ── */
.panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
}
.panel-dark {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
}

/* ── Stat chips ── */
.stat-chip {
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #1d4ed8;
    margin: 2px;
}

/* ── Generate button ── */
#gen-btn { background: linear-gradient(135deg,#3b82f6,#6366f1) !important; color: #fff !important; font-size: 1.05rem !important; border-radius: 10px !important; }
#gen-btn:hover { background: linear-gradient(135deg,#2563eb,#4f46e5) !important; }

/* ── Status bar ── */
#status-box textarea { font-family: monospace; font-size: 0.85rem; color: #1e40af; background: #eff6ff; border-radius: 8px; }

/* ── History table ── */
.history-table { font-size: 0.82rem; }

/* ── Tips card ── */
.tips { background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 0 8px 8px 0; padding: 14px 18px; margin-top: 8px; }
.tips h4 { margin: 0 0 6px; color: #15803d; }
.tips li { color: #166534; margin: 3px 0; }
"""

with gr.Blocks(title="Voice Cloning Studio v2", css=CSS, theme=gr.themes.Soft()) as demo:

    # ── Header ──
    gr.HTML("""
    <div class="studio-header">
      <h1>🎙️ Voice Cloning Studio <span style="font-size:1rem;font-weight:400;color:#90cdf4;">v2</span></h1>
      <p>High-quality AI voice generation · Voice cloning · Emotions · Unlimited text length · Named profiles · History</p>
    </div>
    """)

    with gr.Tabs():

        # ═══════════════════════════════════════════════════
        # TAB 1 — Generate
        # ═══════════════════════════════════════════════════
        with gr.Tab("🎙️ Generate"):
            with gr.Row(equal_height=False):

                # ── Left column: text + controls ──
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Text to Speak",
                        placeholder=(
                            "Enter any amount of text here.\n"
                            "Long texts are automatically chunked and stitched — "
                            "20-30 min audio supported with no extra steps.\n\n"
                            "Tips: Use punctuation (! ? . …) for natural pauses. "
                            "Paragraph breaks add a brief silence."
                        ),
                        lines=10,
                        max_lines=200,
                    )

                    dur_estimate = gr.Textbox(
                        label="⏱ Estimated Duration",
                        value="—",
                        interactive=False,
                        info="Updates as you type. Actual duration may vary ±10%.",
                    )

                    with gr.Row():
                        language_dd = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="English",
                            label="Language",
                            scale=1,
                        )
                        emotion_dd = gr.Dropdown(
                            choices=EMOTIONS,
                            value="neutral",
                            label="Emotion (cloning mode)",
                            scale=1,
                        )

                    with gr.Row():
                        speed_sl = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                            label="Speed  (0.5 = slow · 1.0 = normal · 2.0 = fast)",
                        )
                        temp_sl = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.7, step=0.05,
                            label="Expressiveness / Temperature",
                            info="Higher = more varied, emotional delivery",
                        )

                    with gr.Row():
                        fmt_radio = gr.Radio(
                            choices=FORMATS, value="WAV",
                            label="Output Format",
                        )
                        filename_box = gr.Textbox(
                            label="Custom Filename (optional)",
                            placeholder="e.g.  my_podcast_ep1",
                            info="Leave blank to auto-name with timestamp",
                        )

                    gen_btn    = gr.Button("🎙️ Generate Audio", variant="primary", elem_id="gen-btn")
                    status_box = gr.Textbox(label="Status", interactive=False, elem_id="status-box")

                # ── Right column: voice source ──
                with gr.Column(scale=2):
                    gr.Markdown("### 🔊 Voice Source")

                    with gr.Tabs():
                        with gr.Tab("Upload Sample"):
                            speaker_audio = gr.Audio(
                                label="Voice Sample (WAV / MP3)",
                                type="numpy",
                                sources=["upload"],   # ← BUG FIX: was source= (deprecated)
                                format="wav",
                            )
                            gr.HTML("""
                            <div class="tips">
                              <h4>Best cloning results:</h4>
                              <ul>
                                <li>6–30 seconds of clear speech</li>
                                <li>Single speaker, no background noise</li>
                                <li>WAV or MP3 — sample is auto-denoised</li>
                                <li>Longer = better clone quality</li>
                              </ul>
                            </div>
                            """)

                        with gr.Tab("Saved Profile"):
                            profile_dd = gr.Dropdown(
                                choices=_profile_choices(),
                                value="(none)",
                                label="Select Voice Profile",
                                info="Save profiles in the Profiles tab",
                            )
                            refresh_btn = gr.Button("🔄 Refresh", size="sm")

                    audio_output = gr.Audio(
                        label="Generated Audio",
                        type="filepath",
                    )

            # ── Live duration estimate ──
            text_input.change(estimate_duration, [text_input, speed_sl], dur_estimate)
            speed_sl.change(estimate_duration, [text_input, speed_sl], dur_estimate)

            # ── Generate ──
            gen_btn.click(
                fn=generate_voice,
                inputs=[
                    text_input, speaker_audio, profile_dd,
                    language_dd, emotion_dd, speed_sl, temp_sl,
                    fmt_radio, filename_box,
                ],
                outputs=[audio_output, status_box, gr.State()],
            )
            refresh_btn.click(refresh_profiles, outputs=profile_dd)

        # ═══════════════════════════════════════════════════
        # TAB 2 — Voice Profiles
        # ═══════════════════════════════════════════════════
        with gr.Tab("👤 Voice Profiles"):
            gr.Markdown("""
            Save voice samples under a name so you don't have to re-upload every session.
            Saved profiles are denoised and normalised automatically.
            """)

            with gr.Row():
                with gr.Column():
                    prof_audio_upload = gr.Audio(
                        label="Voice Sample to Save",
                        type="numpy",
                        sources=["upload"],
                        format="wav",
                    )
                    prof_name_box = gr.Textbox(
                        label="Profile Name",
                        placeholder="e.g.  alice  /  podcast_host  /  narrator",
                    )
                    with gr.Row():
                        save_prof_btn   = gr.Button("💾 Save Profile", variant="primary")
                        delete_prof_btn = gr.Button("🗑️ Delete Profile", variant="stop")
                    prof_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column():
                    existing_profiles = gr.Dropdown(
                        choices=_profile_choices(),
                        label="Existing Profiles",
                        value="(none)",
                        info="Select a profile to delete it",
                    )
                    refresh_prof_btn = gr.Button("🔄 Refresh list", size="sm")
                    gr.HTML("""
                    <div class="tips" style="margin-top:16px;">
                      <h4>About Voice Profiles</h4>
                      <ul>
                        <li>Profiles are stored in <code>voice_profiles/</code></li>
                        <li>Each profile is denoised + normalised on save</li>
                        <li>Select a profile in the Generate tab to use it</li>
                        <li>Delete removes the file permanently</li>
                      </ul>
                    </div>
                    """)

            save_prof_btn.click(
                save_profile,
                [prof_audio_upload, prof_name_box],
                [prof_status, existing_profiles],
            )
            delete_prof_btn.click(
                delete_profile,
                [existing_profiles],
                [prof_status, existing_profiles],
            )
            refresh_prof_btn.click(
                lambda: gr.update(choices=_profile_choices()),
                outputs=existing_profiles,
            )

        # ═══════════════════════════════════════════════════
        # TAB 3 — History
        # ═══════════════════════════════════════════════════
        with gr.Tab("📋 History"):
            gr.Markdown("Last 50 generations. Files are saved in the `output/` folder.")
            refresh_hist_btn = gr.Button("🔄 Refresh")
            history_table = gr.Dataframe(
                headers=["Timestamp", "Mode", "Language", "Emotion", "Duration", "File"],
                value=_history_table(),
                elem_classes=["history-table"],
                interactive=False,
                wrap=True,
            )
            refresh_hist_btn.click(lambda: _history_table(), outputs=history_table)

        # ═══════════════════════════════════════════════════
        # TAB 4 — Help
        # ═══════════════════════════════════════════════════
        with gr.Tab("❓ Help"):
            gr.Markdown("""
## How to use Voice Cloning Studio v2

### Basic TTS (no voice sample)
1. Type or paste any text in the **Text to Speak** box — no length limit
2. Choose a **Language**
3. Adjust **Speed** and **Expressiveness** if desired
4. Pick **WAV** or **MP3** output
5. Click **Generate Audio**

### Voice Cloning
1. Upload a 6–30 second voice sample (Upload Sample tab) **or** select a saved profile
2. Choose an **Emotion** style
3. Click **Generate Audio** — the AI will speak in the uploaded voice

### Long-form audio (20–30 min+)
- Just paste your full script — no need to split it manually
- The engine automatically chunks text into ~200-word segments, synthesises each, and crossfade-stitches them into one seamless file
- Watch the **Estimated Duration** field update as you type
- On CPU expect ~1–2 min synthesis time per minute of audio; on GPU ~5–10× faster

### Saving Voice Profiles
- Go to the **Voice Profiles** tab
- Upload a sample and give it a name → **Save Profile**
- On the Generate tab, switch to the **Saved Profile** sub-tab and select it
- Profiles persist across restarts

### Emotion Guide

| Emotion  | Effect                          |
|----------|---------------------------------|
| neutral  | Natural balanced delivery       |
| happy    | Warmer, slightly faster         |
| sad      | Softer, slower                  |
| angry    | Forceful, faster                |
| excited  | Energetic, adds exclamation     |
| whisper  | Quiet, slow, intimate           |

### Supported Languages
English · Spanish · French · German · Italian · Portuguese · Polish · Turkish · Russian · Dutch · Czech · Arabic · Chinese · Japanese · Hungarian · Korean

### Hardware
| Setup | Speed | Notes |
|-------|-------|-------|
| CPU   | ~1–2 min/min audio | Fine for testing |
| GPU (4 GB+ VRAM) | ~5–10 sec/min audio | Set `gpu=True` in app.py |

### First run
The model (~1.5 GB) downloads automatically on first use. Subsequent runs load from cache.
            """)

if __name__ == "__main__":
    demo.launch(share=True)
