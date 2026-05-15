import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

# App setup
st.set_page_config(page_title="PDF Q&A with Gemini")
st.title("📄 PDF Q&A with Gemini")

# Gemini API setup
try:
    genai.configure(api_key=st.secrets["Gemini API Key"])
    st.sidebar.success("✔️ Gemini API Connected")
except:
    st.error("Gemini API Key not found")
    st.stop()


# Chunking function
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file:

    try:
        # Read PDF
        reader = PdfReader(uploaded_file)

        pages_data = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if text:
                pages_data.append({
                    "text": text,
                    "page": page_num + 1
                })

        # Create chunks
        texts = []
        page_numbers = []

        for page_data in pages_data:
            page_chunks = chunk_text(
                page_data["text"]
            )

            for chunk in page_chunks:
                texts.append(chunk)
                page_numbers.append(
                    page_data["page"]
                )

        # Embeddings
        with st.spinner(
            "Creating embeddings..."
        ):

            model_emb = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            embeddings = model_emb.encode(
                texts
            ).astype("float32")

            index = faiss.IndexFlatL2(
                embeddings.shape[1]
            )

            index.add(embeddings)

        st.success(
            f"PDF processed successfully! {len(texts)} chunks created."
        )

        # Ask question
        question = st.text_input(
            "Ask a question about your PDF"
        )

        if st.button("Get Answer"):

            if question.strip() == "":
                st.warning(
                    "Please enter a question"
                )

            else:
                q_emb = model_emb.encode(
                    [question]
                ).astype("float32")

                distances, indices = index.search(
                    q_emb, k=4
                )

                context = "\n\n".join(
                    [texts[i] for i in indices[0]]
                )

                prompt = f"""
Use only the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

                llm = genai.GenerativeModel(
                    "gemini-2.5-flash-lite"
                )

                with st.spinner(
                    "Gemini is thinking..."
                ):
                    response = llm.generate_content(
                        prompt
                    )

                # Answer
                st.subheader("Answer")
                st.write(response.text)

                # Source Pages
                source_pages = sorted(
                    list(
                        set(
                            [
                                page_numbers[i]
                                for i in indices[0]
                            ]
                        )
                    )
                )

                st.write(
                    "📌 Source Pages:",
                    ", ".join(
                        map(str, source_pages)
                    )
                )

                # Context
                with st.expander(
                    "Context Used"
                ):
                    st.write(context)

    except Exception as e:
        st.error(f"Error: {e}")
