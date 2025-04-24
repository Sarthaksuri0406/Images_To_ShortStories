import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8s.pt')

@st.cache_resource
def load_story_model():
    model_path = "./Model"
    tokenizer_path = "./Model"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

yolo_model = load_yolo_model()
story_model, tokenizer, device = load_story_model()

def get_keywords_from_image(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = yolo_model(image_cv)
    detected_objects = results[0].names
    labels = results[0].boxes.cls.cpu().numpy()
    unique_labels = list(set([detected_objects[int(i)] for i in labels]))
    keywords = unique_labels[:2] if unique_labels else ["unknown"]
    return keywords

def generate_story(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True).to(device)
    with torch.no_grad():
        outputs = story_model.generate(
            inputs["input_ids"],
            max_length=400,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=1
        )
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

st.title("Image-to-Story Generator with Object Detection")

num_images = st.number_input("How many images do you want to upload?", min_value=1, max_value=10, step=1)

uploaded_images = []

for i in range(num_images):
    st.markdown(f"Upload Image {i+1}")
    uploaded_file = st.file_uploader(f"Choose an image:", type=["jpg", "jpeg", "png"], key=f"image_{i}")
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption=f"Image {i+1}", use_column_width=True)
        uploaded_images.append(img)

all_keywords=[]
word_list = [
    "apple", "banana", "book", "table", "computer", "dog", "cat", "mountain", "ocean", "planet",
    "sunshine", "cloud", "tree", "flower", "grass", "river", "lake", "forest", "city", "village",
    "star", "moon", "dream", "love", "hope", "family", "friend", "journey", "peace", "harmony",
    "adventure", "challenge", "courage", "strength", "wisdom", "knowledge", "imagination", "creation",
    "freedom", "success", "happiness", "health", "music", "art"
]

if st.button("Generate Stories"):
    if not uploaded_images:
        st.warning("Please upload at least one image.")
    else:
        for idx, img in enumerate(uploaded_images):
            keywords = get_keywords_from_image(img)
            all_keywords.extend(keywords) 
            unique_keywords = list(set(all_keywords))
            prompt = " ".join(unique_keywords)
        prompt = prompt + " " + " ".join(random.sample(word_list, min(len(word_list), 1)))
        print(prompt)
        story = generate_story(prompt)
        st.subheader("Generated Stories")
        st.markdown(f"**Generated Story:** {story}")
