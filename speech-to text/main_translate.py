import os
import tempfile
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

# Load environment variable
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Streamlit setup
st.set_page_config(page_title="🌍 Audio Language Auto-Translation", page_icon="🧠")
st.title("🌍 Auto Language Detection & Translation")
st.markdown("Upload an audio file in any language. It will be transcribed and translated to English.")

# Upload audio file
uploaded_audio = st.file_uploader("📤 Upload audio file", type=["wav", "mp3", "m4a"])

if uploaded_audio:
    with st.spinner("Transcribing and translating..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
                tmp.write(uploaded_audio.read())
                tmp_path = tmp.name

            # Step 1: Transcribe with Whisper (auto language detection)
            with open(tmp_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3-turbo",  # Turbo for faster performance
                    prompt="Transcribe clearly.",
                    response_format="json",
                    temperature=0.5
                )

            detected_text = transcription.text

            # Step 2: Translate to English using LLaMA
            translation_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate the following text into fluent English."},
                    {"role": "user", "content": detected_text}
                ],
                temperature=0.3
            )

            translated_text = translation_response.choices[0].message.content

            # Show results
            st.subheader("📝 Detected Language Transcription")
            st.info(detected_text)

            st.subheader("🌐 Translated to English")
            st.success(translated_text)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
