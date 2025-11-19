import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---- CONFIG ----
INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/metadata.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ---- LOAD EMBEDDINGS + MODEL ----
@st.cache_resource
def load_index_and_model():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return index, metadata, model

# ---- SEARCH FUNCTION ----
def retrieve_docs(query, index, metadata, model, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, top_k)
    retrieved = [metadata[i] for i in I[0]]
    return retrieved

# ---- LLM RESPONSE ----
def generate_answer(query, retrieved_docs):
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])
    prompt = f"""
You are a helpful AI assistant. Use the following context to answer the user question.
If the answer is not in the context, say "I don't have enough information from the document."

Context:
{context}

Question: {query}

Answer:
"""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    # Safely extract the returned text/content from the response to avoid attribute errors
    content = None
    try:
        # Try attribute-style access first
        choices = getattr(response, "choices", None)
        if not choices and isinstance(response, dict):
            choices = response.get("choices")

        if choices and len(choices) > 0:
            choice = choices[0]
            # choice may be object-like or dict-like
            msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
            if msg:
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            else:
                # fallback: older/alternate formats may have 'text' directly on the choice
                content = getattr(choice, "text", None) or (choice.get("text") if isinstance(choice, dict) else None)
    except Exception:
        content = None

    if not content:
        return "I don't have enough information from the document."
    return content.strip()

# ---- STREAMLIT UI ----
st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("📚 RAG Chatbot")
st.caption("Ask questions based on your ingested PDF document")

query = st.text_input("Enter your question here:")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Ask") and query:
    index, metadata, model = load_index_and_model()
    retrieved_docs = retrieve_docs(query, index, metadata, model)
    answer = generate_answer(query, retrieved_docs)

    st.session_state.history.append({"query": query, "answer": answer})

# Display conversation history
for item in reversed(st.session_state.history):
    st.markdown(f"**You:** {item['query']}")
    st.markdown(f"**Bot:** {item['answer']}")
    st.markdown("---")
