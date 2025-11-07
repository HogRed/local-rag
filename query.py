# Imports

import os
from langchain_community.chat_models import ChatOllama                 # Wrapper to use Ollama for local LLM chat
from langchain_core.output_parsers import StrOutputParser               # Converts model responses into plain strings
from langchain_core.runnables import RunnablePassthrough                # Used to pass variables directly into chains
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate   # Tools for creating structured prompts
from langchain_classic.retrievers import MultiQueryRetriever            # Expands a single query into multiple related ones
from get_vector_db import get_vector_db                                 # Connects to ChromaDB for vector retrieval

# Configuration

# Specifies which Ollama model to use for answering questions.
# Example: 'mistral', 'llama2', or any other model available in Ollama.
LLM_MODEL = os.getenv('LLM_MODEL', 'mistral')


# Helper function: get_prompt()
# Purpose:
#   Defines two prompt templates:
#     1. QUERY_PROMPT — used to generate multiple alternative versions of the question
#        (to improve retrieval diversity in the vector search step).
#     2. prompt — used to guide the final answering step, grounded only in retrieved context.
#
# Why this matters:
#   The MultiQueryRetriever uses the first prompt to produce semantically related
#   rephrasings of the user’s question. This helps overcome limitations of similarity
#   search, which can miss relevant chunks if the wording differs.
#   The second prompt then forms the “answering” phase — telling the LLM to use only
#   the retrieved documents to answer the question.

def get_prompt():
    # Prompt for generating alternative search questions
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Prompt for final answer generation
    # The model is instructed to use ONLY the context retrieved from ChromaDB.
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    # Build a structured chat-style prompt template (used by ChatOllama)
    prompt = ChatPromptTemplate.from_template(template)

    return QUERY_PROMPT, prompt


# Main function: query()
# Purpose:
#   Handles the full retrieval and response generation process:
#     1. Initialize the model (ChatOllama).
#     2. Retrieve relevant chunks from ChromaDB using a multi-query approach.
#     3. Pass the retrieved context and question into a prompt chain.
#     4. Generate and return a final answer.
#
# Called by:
#   app.py → route_query()

def query(input):
    # Proceed only if the user provided a valid query string
    if input:
        # Step 1: Initialize the local language model
        # ChatOllama provides a conversational interface to models hosted by Ollama.
        # This allows full offline inference through a locally running model like Mistral.
        llm = ChatOllama(model=LLM_MODEL)

        # Step 2: Connect to the Chroma vector database
        # This gives us access to stored embeddings and similarity search functionality.
        db = get_vector_db()

        # Step 3: Load both the query-generation and answer-generation prompts
        QUERY_PROMPT, prompt = get_prompt()

        # Step 4: Create a retriever that generates multiple query variants
        # MultiQueryRetriever uses the language model to rewrite the input query in
        # several ways before running similarity search. This helps capture semantically
        # related chunks that might otherwise be missed.
        retriever = MultiQueryRetriever.from_llm(
            db.as_retriever(),  # Base retriever (Chroma similarity search)
            llm,                # The LLM used to generate reworded queries
            prompt=QUERY_PROMPT  # The prompt guiding rephrasing
        )

        # Step 5: Build the LangChain "Runnable" pipeline (also called a chain)
        # This defines how data flows through the retrieval and generation process:
        #
        #   User question --> retriever --> context --> prompt --> LLM --> text output
        #
        # The pipe operator (|) is used to connect components together.
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}  # Feed both context and question
            | prompt                                                   # Format input into a final structured prompt
            | llm                                                      # Generate the model’s response
            | StrOutputParser()                                        # Convert the result into a plain text string
        )

        # Step 6: Execute the full chain on the given input
        response = chain.invoke(input)

        # Return the final answer text
        return response

    # Return None if no input was provided
    return None

'''
Notes:
Retrieval-Augmented Generation in action

  - Retrieval (via Chroma) supplies the model with relevant facts.

  - Generation (via ChatOllama) produces a coherent answer grounded in those facts.

MultiQueryRetriever’s role

  - Traditional vector search finds embeddings close to one query.

  - Multi-query expansion improves recall by searching with several paraphrased versions of the question.

  - This helps overcome the “semantic gap” between user phrasing and stored text.

Prompt chaining

  - LangChain’s Runnable design makes it easy to define a linear data pipeline.

  - It models how data flows: question → retrieval → prompt → model → string.

Offline AI workflow

  - Ollama handles both embedding and generation locally — meaning you can run a complete RAG stack without cloud access.
'''