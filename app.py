"""
Voice Cloning Studio - Web UI
Run: python app.py
"""
import os
import gradio as gr
from voice_engine import VoiceEngine

# Initialize engine (will download model on first run)
print("Initializing Voice Engine...")
engine = VoiceEngine(gpu=False)

LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr",
    "Russian": "ru", "Dutch": "nl", "Czech": "cs", "Arabic": "ar",
    "Chinese": "zh", "Japanese": "ja", "Hungarian": "hu", "Korean": "ko"
}

EMOTIONS = ["neutral", "happy", "sad", "angry", "excited", "whisper"]

def generate_voice(text, speaker_audio, language, emotion, speed):
    """Main generation function for Gradio."""
    if not text or text.strip() == "":
        return None, "Please enter some text!"

    lang_code = LANGUAGES.get(language, "en")

    # Save uploaded audio temporarily if provided
    speaker_path = None
    if speaker_audio is not None:
        speaker_path = "samples/uploaded_voice.wav"
        os.makedirs("samples", exist_ok=True)
        # Gradio audio comes as tuple (sample_rate, numpy_array)
        import scipy.io.wavfile as wavfile
        wavfile.write(speaker_path, speaker_audio[0], speaker_audio[1])

    output_path = "output/gradio_generated.wav"
    os.makedirs("output", exist_ok=True)

    try:
        if speaker_path and os.path.exists(speaker_path):
            # Voice cloning with emotion
            result = engine.clone_with_emotion(
                text=text,
                speaker_wav=speaker_path,
                emotion_hint=emotion,
                language=lang_code,
                output_path=output_path
            )
            mode = "Voice Cloning"
        else:
            # Default voice generation
            result = engine.generate_speech(
                text=text,
                language=lang_code,
                speed=speed,
                output_path=output_path
            )
            mode = "Default Voice"

        return result, f"✅ Success! Mode: {mode} | Language: {language} | Emotion: {emotion}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Build Gradio Interface
with gr.Blocks(title="Voice Cloning Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Voice Cloning Studio
    ### High-Quality AI Voice Generation & Voice Cloning with Emotions

    **How to use:**
    1. **Basic TTS**: Type text, select language, and click Generate
    2. **Voice Cloning**: Upload a 6-30 second voice sample (WAV/MP3), type text, and the AI will speak in that voice
    3. **Emotions**: When cloning, select an emotion style to adjust the tone

    *Powered by Coqui XTTS v2*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text to Speak",
                placeholder="Enter text here... Try: 'Hello! This is amazing!'",
                lines=4
            )

            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="Language"
                )
                emotion_dropdown = gr.Dropdown(
                    choices=EMOTIONS,
                    value="neutral",
                    label="Emotion Style (for cloning)"
                )

            speed_slider = gr.Slider(
                minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                label="Speed (for default voice)"
            )

            generate_btn = gr.Button("🎙️ Generate Voice", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            speaker_audio = gr.Audio(
                label="Voice Sample (for cloning)",
                type="numpy",
                source="upload",
                format="wav"
            )
            gr.Markdown("""
            **For best cloning results:**
            - Upload 6-30 seconds of clear speech
            - Single speaker, no background noise
            - WAV or MP3 format
            - The longer the sample, the better the clone
            """)

    audio_output = gr.Audio(label="Generated Audio", type="filepath")

    generate_btn.click(
        fn=generate_voice,
        inputs=[text_input, speaker_audio, language_dropdown, emotion_dropdown, speed_slider],
        outputs=[audio_output, status_text]
    )

    gr.Markdown("""
    ---
    **Tips for better results:**
    - Use punctuation (! ? . ...) to guide natural pauses
    - For emotional speech, the reference audio's emotion matters most
    - First generation takes 2-3 minutes (model download)
    - CPU works but GPU is 10x faster
    """)

if __name__ == "__main__":
    demo.launch(share=True)
