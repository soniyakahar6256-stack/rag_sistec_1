import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader

st.set_page_config(page_title="PDF Q&A with Gemini")

st.title("📄 PDF Q&A with Gemini")

try:
    genai.configure(api_key=st.secrets["Gemini API Key"])
    st.sidebar.success("✅ Gemini API Connected")

except Exception as e:
    st.error("❌ Gemini API Key not found")
    st.stop()

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

try:
    with st.spinner("Extracting text from PDF..."):
        reader = PdfReader(tmp_path)

        pages_data = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if text:
                pages_data.append({
                    "text": text,
                    "page": page_num + 1
                })

    def chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks

    texts = []
    page_numbers = []

    for page_data in pages_data:
        page_chunks = chunk_text(page_data["text"])

        for chunk in page_chunks:
            texts.append(chunk)
            page_numbers.append(page_data["page"])

except Exception as e:
    st.error(f"Error: {e}")

finally:
    os.unlink(tmp_path)

        
