# Local RAG App (Flask + ChromaDB + Ollama)

This project demonstrates a **local Retrieval-Augmented Generation (RAG)** pipeline using **Flask**, **ChromaDB**, and **LangChain**.  
It allows you to upload PDF documents, embed them into a vector database, and query them locally using a language model (e.g., **Ollama**).

---

## Features

- 🧩 **Embeddings & Storage**: Uses ChromaDB as a local vector database.  
- 💬 **Query Interface**: Ask natural-language questions over your PDFs.  
- ⚙️ **LLM Integration**: Connects with Ollama for answers.  
- 🧱 **Modular Design**: Clean separation between embedding, querying, and app logic.

---

## Project Structure
local-rag/

├── app.py # Flask app entrypoint

├── embed.py # Handles PDF embedding

├── query.py # Query logic using LangChain

├── get_vector_db.py # ChromaDB initialization

├── docs/ # Folder for uploaded or sample PDFs

├── chroma/ # Vector database (ignored by Git)

├── venv/ # Virtual environment (ignored)

├── .env # API keys and configuration (ignored)

├── requirements.txt # Dependencies

├── .gitignore # Ignore unnecessary or sensitive files

└── README.md # This file

---

## Setup Instructions

### 1. Clone the Repository
git clone https://github.com/HogRed/local-rag.git

cd local-rag

### 2. Create and Activate a Virtual Environment
python -m venv venv

venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Create a .env File

#### Temporary folder for uploads and processing
TEMP_FOLDER=./_temp

#### Path to store your ChromaDB database
CHROMA_PATH=chroma

#### Name of the Chroma collection
COLLECTION_NAME=local-rag

#### LLM to use for query generation (e.g., Ollama model name)
LLM_MODEL=mistral

#### Embedding model to use for document vectors
TEXT_EMBEDDING_MODEL=nomic-embed-text

## Run the App
python app.py

## Upload and Embed a PDF
Open another terminal (with your venv activated) and run:
curl -X POST "http://127.0.0.1:8080/embed" -H "Content-Type: multipart/form-data" -F "file=@docs/test.pdf"

Expected output:
{"message": "File embedded successfully"}

## Query Your Documents
Once embeddings are created, query your PDF using:
curl -X POST "http://127.0.0.1:8080/query" -H "Content-Type: application/json" -d "{ \"query\": \"Ask something that should be answerable from the PDF\" }"

Expected output:
{"message": "LLM-generated response"}

---

## Credits
Based on [Nasser Maronie’s Dev.to]('https://dev.to/nassermaronie/build-your-own-rag-app-a-step-by-step-guide-to-setup-llm-locally-using-ollama-python-and-chromadb-b12') tutorial.
Customized and updated for modern LangChain, ChromaDB, and Flask.
