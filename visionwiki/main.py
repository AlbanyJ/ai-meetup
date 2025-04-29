from groq import Groq
from dotenv import load_dotenv
import os
import ghana_nlp
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import base64

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key, timeout=60)

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)


# Optional: replace this with your actual vision model
def mock_image_description(image):
     # Call the Groq API to get a response
    response = client.chat.completions.create(
        messages=[
            {
            "role": "system",
            "content": "you are a helpful assistant."
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": "Describe the image in detail.",
        }
        ],
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        #stop=None,
        #stream=False,
    )
    
    return response.choices[0].message.content


st.set_page_config(page_title="🧠 Vision Tool", page_icon="🖼️")
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🧠 Vision Tool</h1>
    <h4 style='text-align: center;'>Upload or paste an image URL to get a description.</h4>
    <br>
    """,
    unsafe_allow_html=True
)

# Upload or URL
upload_tab, url_tab = st.tabs(["📁 Upload Image", "🔗 Image URL"])

image = None

with upload_tab:
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

with url_tab:
    image_url = st.text_input("Paste image URL here:")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Couldn't load image from URL: {e}")

if image:
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Describe Image"):
        with st.spinner("Analyzing image..."):
            # Replace this with actual vision model call
            description = mock_image_description(image)

            st.markdown(f"### 📝 Description:\n{description}")
