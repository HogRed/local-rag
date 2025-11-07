# Import required libraries

import os                            # Provides file system and environment variable utilities
from dotenv import load_dotenv       # Loads variables from a .env file into the environment

# Load environment variables (like TEMP_FOLDER, model names, etc.)
load_dotenv()

from flask import Flask, request, jsonify   # Flask is the lightweight web framework for creating APIs

# Import custom modules from this project
# Each of these scripts defines specific functionality for embedding, querying, and vector DB setup
from embed import embed
from query import query
from get_vector_db import get_vector_db

# Setup and configuration

# TEMP_FOLDER is used to store temporary uploads during the embedding process.
# If not specified in the .env file, it defaults to './_temp'
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')

# Ensure that the temporary folder actually exists on the file system.
# 'exist_ok=True' means it won’t raise an error if the folder already exists.
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Create the Flask application object.
# This acts as the main web server for local RAG system.
app = Flask(__name__)

# API ROUTE 1: Embed endpoint
# Purpose: Accepts a PDF file, processes it into embeddings,
# and stores those embeddings into ChromaDB.

@app.route('/embed', methods=['POST'])
def route_embed():
    # Check if a file was actually sent in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # Check if the user submitted a file with no name
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Call the custom 'embed' function defined in embed.py
    # This function handles reading the PDF, chunking text, creating embeddings,
    # and storing them into the local Chroma vector database.
    embedded = embed(file)

    # Return success or error messages depending on whether the embedding worked
    if embedded:
        return jsonify({"message": "File embedded successfully"}), 200

    return jsonify({"error": "File embedded unsuccessfully"}), 400


# API ROUTE 2: Query endpoint
# Purpose: Accepts a natural language question, searches the
# vector database for relevant context, and passes it to the LLM for an answer.

@app.route('/query', methods=['POST'])
def route_query():
    # Extract JSON data from the incoming POST request
    data = request.get_json()

    # Call the custom 'query' function defined in query.py
    # It handles retrieving relevant chunks from ChromaDB
    # and prompting the local model (e.g., Mistral) for a response.
    response = query(data.get('query'))

    # If the LLM returns a response, return it as JSON
    if response:
        return jsonify({"message": response}), 200

    # If something failed (e.g., no embeddings or model issues), send an error response
    return jsonify({"error": "Something went wrong"}), 400


# Run the Flask app
# host="0.0.0.0" means it’s accessible from any device on your local network.
# debug=True enables live reload and helpful error messages for development.
# Port 8080 is the same endpoint used in curl examples.

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)


'''
Notes:

Flask routes act like functions mapped to URLs.

embed() and query() are decoupled — the API is only responsible for communication, not logic.

The structure mirrors real-world API design patterns (request validation → core logic → structured JSON response).

Running the app with debug=True helps you see stack traces in real time.
'''