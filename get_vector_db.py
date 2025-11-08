# Imports

import os
from langchain_community.embeddings import OllamaEmbeddings       # Interface to generate text embeddings using Ollama
from langchain_community.vectorstores.chroma import Chroma          # LangChain wrapper for ChromaDB

# Environment Configuration
# These environment variables are defined in the .env file and can be overridden per environment.

# Directory where Chroma will persist its database files.
CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')

# Name of the Chroma collection (similar to a table or index).
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')

# Embedding model to use when generating text embeddings.
# This should match an Ollama model capable of generating embeddings.
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'nomic-embed-text')


# Function: get_vector_db
# Purpose:
#   Initializes and returns a Chroma vector database client.
#   This function ensures that the same configuration is used
#   everywhere the vector DB is accessed (embedding and querying).
#
# Workflow:
#   1. Create an embedding function using Ollama (running locally).
#   2. Initialize a ChromaDB collection with that embedding function.
#   3. Return a Chroma object that can be used to store or query embeddings.
#
# This function is used both in:
#   - embed.py (to add new document embeddings)
#   - query.py (to search for relevant chunks during queries)

def get_vector_db():
    # Step 1: Create an embedding generator using Ollama
    # The embedding model is responsible for converting text into high-dimensional numeric vectors.
    # Setting 'show_progress=True' provides feedback during large embedding operations.
    embedding = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL, show_progress=True)

    # Step 2: Initialize a Chroma database (local vector store)
    # - 'collection_name': logical grouping of vectors (like a table name)
    # - 'persist_directory': location where Chroma stores its data files
    # - 'embedding_function': the embedding engine used for encoding and search
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    # Step 3: Return the Chroma client so other scripts can interact with it
    # This object supports methods like:
    #   - add_documents()   → add new text chunks to the database
    #   - similarity_search(query) → retrieve context for a given question
    #   - persist()         → save updates to disk
    return db
    
'''
Notes:
Separation of Concerns

  - The get_vector_db() function ensures there is a single source of truth for how the database is configured.

  - Any script that calls it will use the same model, directory, and collection name.

ChromaDB as a Vector Store

  - Think of ChromaDB like a “searchable memory.”

  - Instead of words, it stores vectors — long lists of numbers that represent text meaning.

  - During a query, Chroma finds the closest vectors (most similar meanings).

Embeddings via Ollama

  - The embedding model translates text → numbers.

  - Ollama runs this model locally, so nothing is sent to the cloud.

Persistence

  - Chroma’s persist_directory allows the system to “remember” embeddings across runs — it saves to disk like a small database.
'''