from groq import Groq
from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load environment and Groq client
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key, timeout=60)  # Set timeout to prevent errors

# Load image captioning model locally
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Generate caption from image
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Use Groq LLM to describe further
def describe_with_groq(caption):
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Based on this image caption: '{caption}', give a detailed description of what the image might show."}
            ],
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error contacting Groq API: {e}"

# Streamlit UI
st.set_page_config(page_title="🧠 WikiVision", page_icon="🖼️")
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧠 WikiVision</h1>
    <h4 style='text-align: center;'>Upload or paste an image URL to get a description.</h4>
    <br>
""", unsafe_allow_html=True)

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
            try:
                caption = generate_caption(image)
                description = describe_with_groq(caption)
                st.markdown(f"**📝 Caption:** {caption}")
                st.markdown(f"**📜 Detailed Description:**\n{description}")
            except Exception as e:
                st.error(f"Failed to analyze image: {e}")