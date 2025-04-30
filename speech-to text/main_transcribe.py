import os
import json
import tempfile
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

# Load environment variable
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Streamlit UI setup
st.set_page_config(page_title="🗣️ Audio Transcriber", page_icon="🎧")
st.title("🎧 Audio Transcription Tool")
st.markdown("Upload an audio file (.wav or .mp3), transcribe it, and ask questions about the content.")

# File uploader
uploaded_audio = st.file_uploader("📤 Upload audio", type=["wav", "mp3", "m4a"])

if uploaded_audio:
    with st.spinner("Transcribing..."):
        try:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_audio.read())
                tmp_path = tmp.name

            # Transcribe audio
            with open(tmp_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3-turbo",
                    prompt="Transcribe clearly.",
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    language="en",  # Use 'en' for English, 'ar' for Arabic
                    temperature=0.5
                )

            # Display full transcription
            st.subheader("📝 Full Transcription")
            transcript_text = transcription.text
            st.write(transcript_text)

            # Display word-level timestamps (FIXED here)
            if "words" in transcription:
                with st.expander("📌 Timestamps (Word-Level)"):
                    for word in transcription.words:
                        st.write(f"{word.word} — ⏱️ {word.start}s to {word.end}s")

            # Q&A Section
            st.subheader("❓ Ask a question about the audio")
            user_question = st.text_input("Type your question here")

            if st.button("💬 Ask"):
                if not user_question.strip():
                    st.warning("Please enter a question before clicking the button.")
                else:
                    with st.spinner("Thinking..."):
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant. Answer based only on the transcript provided."},
                                {"role": "user", "content": f"Transcript: {transcript_text}"},
                                {"role": "user", "content": user_question}
                            ],
                            temperature=0.5,
                            max_completion_tokens=1024,
                        )
                        st.success("✅ Response:")
                        st.markdown(response.choices[0].message.content)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
