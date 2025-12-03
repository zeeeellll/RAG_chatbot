import streamlit as st
import os
import pickle
import tempfile
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Config
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_PAGES = 3

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model"""
    return SentenceTransformer(EMBED_MODEL_NAME)

def extract_text_from_pdf(pdf_file, max_pages=MAX_PAGES):
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        num_pages = min(len(pdf_reader.pages), max_pages)
        
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        return text, num_pages
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None, 0

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def create_vector_index(chunks, model):
    """Create FAISS index from text chunks"""
    try:
        # Create embeddings
        embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings) # type: ignore
        
        return index
    except Exception as e:
        st.error(f"Error creating index: {str(e)}")
        return None

def search_similar_chunks(query, index, chunks, model, top_k=3):
    """Search for similar chunks using the query"""
    try:
        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = index.search(query_embedding, top_k)
        
        # Get relevant chunks
        relevant_chunks = [chunks[i] for i in indices[0]]
        return relevant_chunks
    except Exception as e:
        st.error(f"Error searching: {str(e)}")
        return []

def generate_response(query, context):
    """Generate response using OpenAI API with RAG context"""
    try:
        client = openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
        
        prompt = f"""Based on the following context from the document, please answer the question.

Context:
{context}

Question: {query}

Please provide a clear and concise answer based only on the information in the context. If the answer cannot be found in the context, say so."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": 
                                        """You are an AI assistant that answers strictly based on the content extracted from the user-provided PDF.
                        You must rely only on the text chunks returned by the retrieval system.
                        If the answer is not present in the PDF, clearly say you cannot find it.

                        RULES:

                        Never invent information. No guessing, no assumptions.

                        Everything must be grounded in the PDF text.

                        If the PDF is unclear, incomplete, or conflicting → point it out directly.

                        If user asks something outside the PDF → respond:
                        “This information is not available in the provided PDF.”

                        Always give concise, direct answers.

                        When the user asks broad questions like “What is this PDF about?”, generate a summary only from its content.

                        When the user asks for details (definitions, dates, steps, explanations), extract them exactly from the PDF text.

                        If multiple sections relate to the question, synthesize them clearly.

                        Use the original meaning and context—do NOT reinterpret.

                        OUTPUT STYLE:

                        Straightforward

                        Fact-based

                        No opinions

                        No hallucinations

                        No filler sentences

                        GOAL:
                        Provide the most accurate, PDF-grounded answer every time."""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    except Exception as e:
        # Fallback to simple context-based response
        return f"Based on the document: {context[:500]}... \n\nTo enable AI responses, please add your OpenAI API key to Streamlit secrets. Error: {str(e)}"

# UI
st.title("🤖 RAG Chatbot with PDF Upload")
st.markdown("Upload a PDF (max 3 pages) and ask questions about its content!")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help=f"Upload a PDF file (maximum {MAX_PAGES} pages)"
    )
    
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                # Extract text
                text, num_pages = extract_text_from_pdf(uploaded_file, MAX_PAGES)
                
                if text:
                    st.success(f"✅ Extracted text from {num_pages} pages")
                    
                    # Chunk text
                    chunks = chunk_text(text)
                    st.info(f"📝 Created {len(chunks)} chunks")
                    
                    # Load model
                    if st.session_state.model is None:
                        st.session_state.model = load_embedding_model()
                    
                    # Create index
                    index = create_vector_index(chunks, st.session_state.model)
                    
                    if index:
                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.success("✅ Vector index created successfully!")
                        st.balloons()
    
    if st.session_state.index is not None:
        st.success("🟢 Document indexed and ready!")
        if st.button("Clear Document"):
            st.session_state.index = None
            st.session_state.chunks = []
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. Upload a PDF (max 3 pages)
    2. Click 'Process PDF'
    3. Ask questions about the content
    4. Get AI-powered answers
    """)

# Main chat interface
if st.session_state.index is None:
    st.info("👆 Please upload and process a PDF document to start chatting!")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "context" in message:
                with st.expander("📚 View source context"):
                    st.text(message["context"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search for relevant chunks
                relevant_chunks = search_similar_chunks(
                    prompt,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.model
                )
                
                # Combine context
                context = "\n\n".join(relevant_chunks)
                
                # Generate response
                response = generate_response(prompt, context)
                
                st.markdown(response)
                
                with st.expander("📚 View source context"):
                    st.text(context)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "context": context
        })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Powered by Sentence Transformers, FAISS, and OpenAI
</div>
""", unsafe_allow_html=True)