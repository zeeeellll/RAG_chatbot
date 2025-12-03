#  RAG Chatbot with PDF Upload

The Chatbot is a Streamlit-based application that allows users to upload PDF documents and interact with them using a Retrieval-Augmented Generation (RAG) system. It leverages Sentence Transformers for embedding generation, FAISS for efficient similarity search, and OpenAI's GPT models for generating conversational responses.

## Features

*   **PDF Upload:** Easily upload PDF documents (up to 3 pages).
*   **Text Extraction:** Extracts text content from uploaded PDFs.
*   **Text Chunking:** Splits the extracted text into manageable, overlapping chunks.
*   **Vector Indexing:** Creates a FAISS vector index for fast semantic search of document chunks.
*   **RAG-powered Chat:** Answers user questions based on the context retrieved from the uploaded PDF using OpenAI's API.
*   **Clear Document:** Option to clear the loaded document and index.

## Technologies Used

*   **Streamlit:** For creating the interactive web application.
*   **Sentence Transformers:** For generating embeddings from text chunks.
*   **FAISS:** For efficient similarity search on the generated embeddings.
*   **PyPDF2:** For extracting text from PDF files.
*   **OpenAI API:** For generating AI-powered responses.
*   **python-dotenv:** For managing environment variables (API keys).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/YPV_chatbot.git
    cd YPV_chatbot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is not available, you can generate it using `pip freeze > requirements.txt` after installing the dependencies manually, or install them one by one: `streamlit`, `sentence-transformers`, `faiss-cpu`, `PyPDF2`, `openai`, `python-dotenv`)*

4.  **Set up OpenAI API Key:**
    Create a `.env` file in the root directory of the project and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open in Browser:**
    The application will open in your web browser, usually at `http://localhost:8501`.

3.  **Upload a PDF:**
    Use the sidebar to upload a PDF file (maximum 3 pages).

4.  **Process PDF:**
    Click the "Process PDF" button to extract text, chunk it, and create a vector index.

5.  **Start Chatting:**
    Once the document is indexed, you can ask questions about its content in the chat interface.
