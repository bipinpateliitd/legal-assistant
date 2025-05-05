#!/usr/bin/env python3
"""
Bhartiya Nyaya Sanhita (BNS) Legal Assistant Web Application

This Streamlit application provides a user-friendly interface for querying and retrieving
information from the Bhartiya Nyaya Sanhita legal code. It uses a hybrid retrieval system
with both dense and sparse embeddings to find relevant legal information based on user queries.

Key features:
- Chat-based interface for asking legal questions
- Automatic detection of section numbers in queries
- Hybrid vector search combining semantic understanding and keyword matching
- Source document display for transparency and verification
- Persistent chat history during the session

The application connects to a Qdrant vector database that stores embeddings of BNS sections
and uses OpenAI's models for embeddings and response generation.
"""

import streamlit as st
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import re
from qdrant_client.http import models

# Load environment variables from .env file
load_dotenv()

# Configure the Streamlit page settings
st.set_page_config(
    page_title="Legal Assistant",  # Browser tab title
    page_icon="⚖️",              # Browser tab icon
    layout="wide"                # Use wide layout for better readability
)

# Set up the main application header and description
st.title("Legal Assistant")
st.subheader("Ask questions about BNS sections")
st.write("""
    This application helps you find information about BNS sections. You can:
    - Ask about specific section numbers (e.g., "What does section 123 say about...") 
    - Ask general questions about legal topics
""")

# Initialize configuration parameters from environment variables
url = os.getenv("QDRANT_CLOUD_URL")           # Qdrant cloud service URL
api_key = os.getenv("QDRANT_API_KEY")         # API key for authentication
embedding_model = "text-embedding-3-large"     # OpenAI embedding model for dense vectors
chat_model = "gpt-4.1-nano"                   # OpenAI chat model for response generation
collection_name = "bns_sections_hybrid"        # Qdrant collection name storing the vectors

# Initialize connection to Qdrant vector database
client = QdrantClient(
    url=url,
    api_key=api_key,
    prefer_grpc=True  # Use gRPC for better performance
)

# Initialize embedding models and language model
dense_embedding = OpenAIEmbeddings(model=embedding_model)  # For semantic understanding
sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")  # For keyword matching
llm = ChatOpenAI(model=chat_model, temperature=0)  # Zero temperature for deterministic responses

# Set up the vector store with hybrid search capabilities
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=dense_embedding,         # For dense vector search
    sparse_embedding=sparse_embedding, # For sparse vector search
    retrieval_mode=RetrievalMode.HYBRID,  # Combine both search methods
    vector_name="dense",               # Name of dense vector field in Qdrant
    sparse_vector_name="sparse",      # Name of sparse vector field in Qdrant
)

def extract_section_numbers(query):
    """
    Extract all section numbers mentioned in a query string.
    
    This function uses regex to find all standalone numbers in the query,
    which are likely to be section references in the legal context.
    
    Args:
        query (str): The user's query text
        
    Returns:
        list[int]: A list of integers representing section numbers found in the query
    """
    return [int(n) for n in re.findall(r"\b(\d+)\b", query)]

def format_docs(docs):
    """
    Format retrieved documents for readable display in the UI.
    
    This function takes the raw document objects returned from the vector store
    and formats them into a readable string, including section numbers and content.
    
    Args:
        docs (list): List of document objects from the vector store
        
    Returns:
        str: A formatted string containing all document contents with section information
    """
    formatted_text = ""
    for i, doc in enumerate(docs):
        # Extract section number from metadata, default to 'N/A' if not found
        section_num = doc.metadata.get('sections', 'N/A')
        # Format each document with an index, section number, and its content
        formatted_text += f"Document {i+1} (Section {section_num}):\n{doc.page_content}\n\n"
    return formatted_text

# Initialize session state for persistent chat history during the session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages in the chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Create the chat input field for user questions
user_input = st.chat_input("Ask your legal question here...")

# Process user input when submitted
if user_input:
    # Display the user's message in the chat
    st.chat_message("user").write(user_input)
    # Add the message to session state to persist it
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Create and display the assistant's response
    with st.chat_message("assistant"):
        # Show a spinner while processing the query
        with st.spinner("Searching for information..."):
            # Step 1: Extract any section numbers from the query
            section_nums = extract_section_numbers(user_input)
            
            # Step 2: Retrieve relevant documents based on the query
            if section_nums:
                # If section numbers are found, show them to the user
                st.info(f"Found section numbers in your query: {section_nums}")
                # Use exact section number matching for precise retrieval
                results = vectorstore.similarity_search(
                    query="",  # Empty query when using exact section filtering
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.sections", 
                                match=models.MatchAny(any=section_nums)
                            )
                        ]
                    )
                )
            else:
                # For general queries without section numbers, use hybrid semantic search
                results = vectorstore.similarity_search(user_input, k=5)  # Retrieve top 5 results
            
            # Step 3: Format the retrieved documents into context for the LLM
            context = format_docs(results)
            
            # Step 4: Create the prompt template for the LLM
            template = """
            You are a helpful legal assistant. Use the following context to answer the question.
            If the answer cannot be found in the context, say so.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
            
            prompt_template = ChatPromptTemplate.from_template(template)
            
            # Step 5: Generate the response using the LLM
            chain = prompt_template | llm  # Create a simple chain: prompt -> LLM
            response = chain.invoke({
                "context": context,
                "question": user_input
            })
            
            # Step 6: Display the generated response
            st.write(response.content)
            # Add the response to session state to persist it
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            
            # Step 7: Show source documents in an expandable section for transparency
            with st.expander("View Source Documents"):
                st.text(context)

# Create a sidebar with additional information about the application
with st.sidebar:
    st.header("About This App")
    st.write("This legal assistant uses AI to help you understand BNS sections.")
    
    st.subheader("Features")
    st.write("- Section-specific search")
    st.write("- Hybrid retrieval (dense + sparse vectors)")
    st.write("- Context-aware responses")
    
    st.subheader("Tips for Better Results")
    st.write("- Be specific in your questions")
    st.write("- Include section numbers when possible")
    st.write("- Check the source documents for reference")