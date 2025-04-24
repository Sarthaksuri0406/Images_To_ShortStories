# Images_To_ShortStories

Turn ordinary images into creative, AI-generated short stories using deep learning and natural language processing.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-enabled-red?logo=pytorch)
![OpenAI](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

**Images to Short Stories** is a deep learning pipeline that:
1. Detects objects in an image using **YOLOv8**.
2. Converts detected elements into a prompt using a custom prompt builder.
3. Generates a short, creative story using a fine-tuned **language model** (e.g., GPT2 or T5).

This app can be used for:
- Educational storytelling for kids
- Creative inspiration
- Accessibility tools
- Visual content enhancement

---

## Project Structure

```bash
Images_To_ShortStories/
├── app_st.py                # Main Streamlit app
├── yolov8s.pt               # YOLOv8 model weights (Git LFS)
├── Model/                   # Fine-tuned language model and tokenizer
├── Final_Notebook.ipynb     # Training + inference pipeline
├── requirements.txt         # All dependencies
└── README.md                # This file
