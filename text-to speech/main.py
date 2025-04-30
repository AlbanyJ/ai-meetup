import os
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

# Load API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Streamlit UI
st.set_page_config(page_title="🗣️ Text to Speech", page_icon="🔊")
st.title("🔊 Text to Speech Generator")

text_input = st.text_area("✏️ Enter text to convert to speech", height=150)
generate = st.button("🎧 Generate Speech")

if generate and text_input.strip():
    with st.spinner("Generating speech..."):
        try:
            # Generate speech
            response = client.audio.speech.create(
                model="playai-tts",
                voice="Fritz-PlayAI",  # You can change voice here
                input=text_input,
                response_format="wav"
            )

            # Save to file
            file_path = "output_speech.wav"
            response.write_to_file(file_path)

            # Play and download
            audio_file = open(file_path, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
            st.download_button("⬇️ Download Audio", data=audio_bytes, file_name="speech.wav")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
else:
    st.caption("Enter some text and press the button to create audio.")
