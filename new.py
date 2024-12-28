import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# Load models
@st.cache_resource  # Cache the embedding model to avoid reloading
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

@st.cache_resource  # Cache the QA model
def load_qa_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

embedding_model = load_embedding_model()
qa_model = load_qa_model()

# PDF Processing: Extract text
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Chunk text into manageable sizes
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Build FAISS Index
def build_faiss_index(text_chunks, embedding_model):
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)  # Convert to numpy array
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Retrieve relevant chunks
def retrieve_relevant_chunks(question, text_chunks, index, embedding_model, top_k=3):
    question_embedding = embedding_model.encode([question], convert_to_numpy=True)
    _, indices = index.search(np.array(question_embedding, dtype="float32"), top_k)
    return [text_chunks[i] for i in indices[0]]

# Answer questions using QA model
def answer_question(question, context, qa_model):
    try:
        response = qa_model(question=question, context=context)
        return response["answer"]
    except Exception as e:
        return f"An error occurred while answering the question: {e}"

# Streamlit Interface
st.title(" TEXTBOOK BASED QUESTION ANSWERING SYSTEM")
st.sidebar.header("Upload PDFs")

uploaded_files = st.sidebar.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        raw_text = extract_text_from_pdfs(uploaded_files)
        if raw_text.strip():
            text_chunks = chunk_text(raw_text)
            faiss_index, embeddings = build_faiss_index(text_chunks, embedding_model)
            st.success("PDFs Processed Successfully!")
        else:
            st.error("No text extracted from the uploaded PDFs. Please upload valid PDF files.")

    question = st.text_input("Ask a Question")
    if question:
        with st.spinner("Retrieving relevant context..."):
            relevant_chunks = retrieve_relevant_chunks(question, text_chunks, faiss_index, embedding_model)
            context = " ".join(relevant_chunks)

        with st.spinner("Generating Answer..."):
            answer = answer_question(question, context, qa_model)
            st.write(f"*Answer:* {answer}")

        with st.expander("Relevant Context"):
            st.write(context)