from groq import Groq
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=api_key)

# Streamlit app UI
st.set_page_config(page_title="VISIONWIKI", page_icon="🌟")  # optional for a better header

st.title("VISIONWIKI")
st.write("**ESSAIE-MOI!**")  # Make it bold

# User input
user_input = st.text_input("TRY ME:")

# When user clicks the button
if st.button("Send"):
    if user_input.strip():  # Check if input is not empty or spaces
        with st.spinner('Thinking...'):  # Spinner while waiting
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ],
                    model="llama-3.3-70b-versatile",  # or whichever correct model you want (yours was 'llama-3.3-70b-versatile', check if that’s accurate)
                    temperature=0.5,
                    max_completion_tokens=1024,
                    top_p=1,
                    stop=None,
                    stream=False,
                )

                # Extract the response
                response = chat_completion.choices[0].message.content

                # Show the response
                st.success(response)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please type a message before sending.")

