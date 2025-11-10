# Local RAG App (Flask + ChromaDB + Ollama)

This project implements a **local Retrieval-Augmented Generation (RAG)** system using **Flask**, **ChromaDB**, and **LangChain**. It allows users to upload PDF files, generate embeddings, store them in a local vector database, and perform natural language queries over the embedded documents. The application runs entirely on local infrastructure, integrating with **Ollama** for model inference.

---

## Overview

This implementation follows the RAG design pattern, where:

1. Documents are embedded using a local embedding model (e.g., `nomic-embed-text`).
2. The embeddings are stored and indexed in **ChromaDB** for efficient similarity search.
3. A query is processed by retrieving the most relevant chunks from the database.
4. The retrieved context is passed to a local language model (e.g., `mistral` via Ollama) to generate a final response.

The result is a fully offline, modular RAG system that can be customized for different document types, models, or database configurations.

---

## Features

* **Local Vector Database**: Uses ChromaDB for embedding storage and retrieval.
* **Model Integration**: Runs inference through Ollama with a configurable LLM.
* **REST API Interface**: Flask provides clean `/embed` and `/query` endpoints.
* **Lightweight & Extensible**: No external dependencies beyond the local environment.
* **Modular Codebase**: Clear separation of application, embedding, and query logic.

---

## Project Structure

```
local-rag/
├── app.py               # Flask app entrypoint
├── embed.py             # Handles PDF embedding
├── query.py             # Query logic using LangChain
├── get_vector_db.py     # ChromaDB initialization
├── docs/                # Local folder for uploaded or sample PDFs
├── chroma/              # Vector database (ignored by Git)
├── venv/                # Virtual environment (ignored)
├── .env                 # Environment configuration (ignored)
├── requirements.txt     # Dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

---

## Setup and Configuration

### 1. Clone the Repository

```bash
git clone https://github.com/HogRed/local-rag.git
cd local-rag
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` in the project root with the following values:

```
TEMP_FOLDER=./_temp
CHROMA_PATH=chroma
COLLECTION_NAME=local-rag
LLM_MODEL=mistral
TEXT_EMBEDDING_MODEL=nomic-embed-text
```

These variables control where temporary files and embeddings are stored, the name of the ChromaDB collection, and which local models are used for embeddings and inference.

---

## Running the Application

### Start the Flask Server

```bash
python app.py
```

Once started, Flask will serve the API locally (by default at `http://127.0.0.1:8080`).

---

## Using the API

### 1. Upload and Embed a Document

Open a new terminal window (Flask must be running) and execute:

```bash
curl -X POST "http://127.0.0.1:8080/embed" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@docs/test.pdf"
```

Successful output:

```json
{"message": "File embedded successfully"}
```

### 2. Query the Vector Database

Once embeddings are created, run:

```bash
curl -X POST "http://127.0.0.1:8080/query" ^
  -H "Content-Type: application/json" ^
  -d "{ \"query\": \"Summarize the document.\" }"
```

Expected response:

```json
{"message": "Generated response from the local model."}
```

---

## Git Workflow

```bash
# Save and commit changes
git add .
git commit -m "Describe your changes"
git push origin main

# Sync with remote updates
git pull origin main
```

---

## Requirements

* Python 3.10 or higher
* Ollama installed locally with the `mistral` model and `nomic-embed-text` embedding model pulled.

  ```bash
  ollama pull mistral
  ollama pull nomic-embed-text
  ```
* ChromaDB and LangChain dependencies (installed via `requirements.txt`)

---

## Credits

Based on [Nasser Maronie’s Dev.to tutorial](https://dev.to/nassermaronie/build-your-own-rag-app-a-step-by-step-guide-to-setup-llm-locally-using-ollama-python-and-chromadb-b12), with updates for current versions of LangChain, ChromaDB, and Flask.
