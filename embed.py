# Imports

import os
from datetime import datetime
from werkzeug.utils import secure_filename                # Helps sanitize uploaded file names
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_vector_db import get_vector_db                   # Custom helper to connect to ChromaDB

# Configuration

# TEMP_FOLDER defines where uploaded files are stored before being processed.
# If it's not set in .env, the default is './_temp'.
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')

# Helper function: File validation
# Checks that the uploaded file has an extension and that it is a PDF.
# This helps prevent users from uploading unsupported file types (e.g., .exe or .txt).

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}


# Helper function: Save uploaded file
# Saves the uploaded PDF temporarily on disk before it’s processed.
# Adds a timestamp prefix to avoid filename collisions.
# The use of 'secure_filename' prevents directory traversal or unsafe file names.

def save_file(file):
    # Generate a unique timestamped filename
    ct = datetime.now()
    ts = ct.timestamp()
    filename = str(ts) + "_" + secure_filename(file.filename)

    # Build the full path inside the temporary directory
    file_path = os.path.join(TEMP_FOLDER, filename)

    # Save the uploaded file to disk
    file.save(file_path)

    # Return the path so it can be loaded later
    return file_path


# Helper function: Load and split PDF into chunks
# This is where the document is converted into smaller sections that can be embedded.
# LangChain’s 'UnstructuredPDFLoader' extracts the text, and
# 'RecursiveCharacterTextSplitter' breaks it into overlapping text chunks.

def load_and_split_data(file_path):
    # Load the raw PDF text into LangChain document objects
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()

    # Split text into overlapping chunks.
    # Overlaps help preserve context across chunk boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=7500,    # Each chunk will contain up to 7500 characters
        chunk_overlap=100   # Each chunk shares 100 characters with the previous one
    )

    # Split and return as a list of LangChain Document objects
    chunks = text_splitter.split_documents(data)
    return chunks


# Main function: Embed the document
# This is the core function that gets called from app.py.
# It performs the following steps:
#   1. Validate the uploaded file.
#   2. Save it temporarily to disk.
#   3. Load and split its contents.
#   4. Add embeddings to the vector database.
#   5. Persist the database and remove the temporary file.

def embed(file):
    # Proceed only if the file exists, has a name, and is a PDF.
    if file.filename != '' and file and allowed_file(file.filename):
        # Save file locally for temporary processing
        file_path = save_file(file)

        # Load PDF text and split into smaller pieces
        chunks = load_and_split_data(file_path)

        # Connect to or create the Chroma vector database
        db = get_vector_db()

        # Add the new document chunks to the vector store
        db.add_documents(chunks)

        # Save (persist) the updated embeddings database to disk
        db.persist()

        # Remove the temporary file now that it has been processed
        os.remove(file_path)

        # Return True to indicate success
        return True

    # If the file didn’t meet requirements or failed processing, return False
    return False
    
'''
Notes:

Chunking:

  - The embedding model can’t process an entire PDF at once.

  - Splitting into chunks allows semantic search to work on smaller, context-rich sections.

Vector storage:

  - Each chunk is turned into a vector (numerical representation).

  - These vectors are stored in ChromaDB, allowing the model to quickly retrieve relevant passages.

Temporary files:

  - PDFs are saved only long enough to process them, then deleted for security and cleanliness.

Separation of concerns:

  - This script does all preprocessing and embedding.

  - The Flask API only calls it—it doesn’t handle any embedding logic itself.
'''