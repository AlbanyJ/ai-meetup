from groq import Groq
from dotenv import load_dotenv

import os
import streamlit as st

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

# Streamlit app user interface
st.title("VISIONWIKI")
st.write("ESSAIE-MOI!")

user_input = st.text_input("TRY ME: ")


if st.button("Send"):
    if user_input:
        chat_completion = client.chat.completions.create(
            #
            # Required parameters
            #
            messages=[
                # Set an optional system message. This sets the behavior of the
                # assistant and can be used to provide specific instructions for
                # how it should behave throughout the conversation.
                {
                    "role": "system",
                    "content": "you are a helpful assistant."
                },
                # Set a user message for the assistant to respond to.
                {
                    "role": "user",
                    "content": user_input,
                }
            ],

            # The language model which will generate the completion.
            model="llama-3.3-70b-versatile",

            #
            # Optional parameters
            #

            # Controls randomness: lowering results in less random completions.
            # As the temperature approaches zero, the model will become deterministic
            # and repetitive.
            temperature=0.5,

            # The maximum number of tokens to generate. Requests can use up to
            # 32,768 tokens shared between prompt and completion.
            max_completion_tokens=1024,

            # Controls diversity via nucleus sampling: 0.5 means half of all
            # likelihood-weighted options are considered.
            top_p=1,

            # A stop sequence is a predefined or user-specified text string that
            # signals an AI to stop generating content, ensuring its responses
            # remain focused and concise. Examples include punctuation marks and
            # markers like "[end]".
            stop=None,

            # If set, partial message deltas will be sent.
            stream=False,
        )

    # Print the completion returned by the LLM.
   # print(chat_completion.choices[0].message.content)

        response = chat_completion.choices[0].message.content
        st.success(response)
    else:
        st.warning("Please type a message before sending.")