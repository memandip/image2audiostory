import os
from dotenv import load_dotenv 
import requests
# import torch
import streamlit as st
# from transformers import pipeline
from huggingface_hub import InferenceClient
from datetime import datetime

load_dotenv()

HUGGINGFACE_API_TOKEN=os.getenv("HUGGINGFACE_API_TOKEN")
client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.2",
    token=HUGGINGFACE_API_TOKEN,
)

print(f"HUGGINGFACE_API_TOKEN: {HUGGINGFACE_API_TOKEN}")

# replace local model with huggingface inference client
# # image to text model
# def image2text(url):
#     image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=1000, device=1)
#     text = image_to_text(url)[0]['generated_text']

#     return text

# image to text model
def image2text(filename):
    print(filename)
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data).json()
    print("\n=====================================================\n")
    print(response)
    print("\n=====================================================\n")
    return response[0]["generated_text"]

# llm
def generate_story(scenario):
    template = """
        You are a storyteller;
        You can generate a story based on a simple narrative under 100 words with the following context:
        CONTEXT: {scenario}
    """.format(scenario=scenario)

    # Inference client load chat completion
    response = client.chat_completion(
        messages=[{"role": "user", "content": template}],
        max_tokens=500,
        # stream=True,
    )

    return response.choices[0].message.content


# text to speech model
def text2speech(text, filename):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    payload = {
        "inputs": text
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    filepath = 'output.flac' if filename is None else filename
    
    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath

def main():

    st.set_page_config(page_title="img 2 audio story", page_icon=None)
    st.header("Turn your image into Audio Story")
    uploaded_file = st.file_uploader("choose an image....", type=["jpg", "png", "jpeg", "JPG", "PNG", "JPEG", "bmp"])

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
            st.image(uploaded_file, caption="Uploaded image..",
                use_column_width=True)
            scenerio = image2text(uploaded_file.name)
            story = generate_story(scenario=scenerio)
            filename = f"{datetime.now()}-story.flac"
            filepath = text2speech(story, filename=filename)

            with st.expander("scenerio"):
                st.write(scenerio)
            with st.expander("story"):
                st.write(story)

            st.audio(filepath)


if __name__ == "__main__":
     main()

# run streamlit run python-script-file-name with extension