import streamlit as st
import boto3
import json
import base64
import os
import random
from PIL import Image
import io

# Initialize Bedrock client
client = boto3.client("bedrock-runtime", region_name="eu-central-1")

def generate_text(model_id, prompt, max_tokens):
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    request = json.dumps(native_request)
    response = client.invoke_model(modelId=model_id, body=request)
    return json.loads(response['body'].read())['content'][0]['text']

def generate_image(model_id, prompt):
    seed = random.randint(0, 4294967295)
    native_request = {
        "text_prompts": [{"text": prompt}],
        "style_preset": "photographic",
        "seed": seed,
        "cfg_scale": 10,
        "steps": 30,
    }
    request = json.dumps(native_request)
    response = client.invoke_model(modelId=model_id, body=request)
    model_response = json.loads(response["body"].read())
    base64_image_data = model_response["artifacts"][0]["base64"]
    return Image.open(io.BytesIO(base64.b64decode(base64_image_data)))

def process_image_with_claude(model_id, image, prompt):
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    base64_image = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                    {"type": "text", "text": prompt}
                ],
            }
        ],
    }
    request = json.dumps(native_request)
    response = client.invoke_model(modelId=model_id, body=request)
    return json.loads(response['body'].read())['content'][0]['text']

st.title("Multi-Model AI Application")

task = st.radio("Select Task", ["Generate Text", "Generate Image", "Process Image"])

if task == "Generate Text":
    model = st.selectbox("Select Text Model", ["Claude 3.5 Sonnet", "Claude 3 Sonnet", "Claude 3 Haiku"])
    model_id_map = {
        "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0"
    }
    prompt = st.text_area("Enter your prompt")
    max_tokens = st.slider("Max Tokens", 1, 1000, 512)
    
    if st.button("Generate Text"):
        with st.spinner("Generating text..."):
            response = generate_text(model_id_map[model], prompt, max_tokens)
            st.text_area("Generated Text", response, height=300)

elif task == "Generate Image":
    model = st.selectbox("Select Image Model", ["Amazon Titan Image Generator", "Stable Diffusion"])
    model_id_map = {
        "Amazon Titan Image Generator": "amazon.titan-image-generator-v1",
        "Stable Diffusion": "stability.stable-diffusion-xl-v1"
    }
    prompt = st.text_input("Enter image prompt")
    
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            image = generate_image(model_id_map[model], prompt)
            st.image(image, caption="Generated Image", use_column_width=True)

elif task == "Process Image":
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    prompt = st.text_input("Enter prompt for image analysis")
    
    if uploaded_file is not None and prompt:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Process Image"):
            with st.spinner("Processing image..."):
                response = process_image_with_claude("anthropic.claude-3-sonnet-20240229-v1:0", image, prompt)
                st.text_area("Claude's Response", response, height=300)

st.sidebar.text("Note: This application requires proper AWS credentials and permissions to use Amazon Bedrock services.")