#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) Evaluation for Legal Assistant

This script evaluates the performance of a RAG system built for the Bhartiya Nyaya Sanhita legal assistant.
It uses a test dataset of legal queries and evaluates the system's ability to retrieve relevant contexts
and generate accurate, faithful responses.

The evaluation uses the RAGAS framework to measure:
1. Context Recall - How well the system retrieves relevant contexts
2. Faithfulness - How well the response stays true to the retrieved contexts
3. Factual Correctness - How accurate the generated responses are

Results are printed at the end showing the system's performance on these metrics.
"""

import os
import re
import pandas as pd
import ast
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_core.prompts import ChatPromptTemplate

# Import RAGAS evaluation framework
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# Load environment variables and test dataset
load_dotenv()
df = pd.read_csv('/home/bipin/Documents/projects/legal_assistant/data/bns_testset.csv')
print(f"Loaded test dataset with {len(df)} queries")

# Initialize Qdrant client, embeddings, and language models
client = QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=True
)

# Initialize embeddings and models
dense_embedding = OpenAIEmbeddings(model="text-embedding-3-large")  # Dense vector embeddings
sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")       # Sparse vector embeddings for hybrid search
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)              # LLM for generating responses
evaluator_llm = LangchainLLMWrapper(llm)                           # Wrapped LLM for evaluation

# Initialize vector store with hybrid search capability
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="bns_sections_hybrid",
    embedding=dense_embedding,
    sparse_embedding=sparse_embedding,
    retrieval_mode=RetrievalMode.HYBRID,  # Use both dense and sparse embeddings
    vector_name="dense",
    sparse_vector_name="sparse",
)

def extract_section_numbers(query: str):
    """
    Extract section numbers from a query string.
    
    Args:
        query (str): The query text to extract section numbers from
        
    Returns:
        list[int]: List of section numbers found in the query
    """
    return [int(n) for n in re.findall(r"\b(\d+)\b", query)]

# Define prompt template for the RAG system
TEMPLATE = """
You are a helpful legal assistant. Use the following context to answer the question.
If the answer cannot be found in the context, say so.

Context:
{context}

Question:
{question}

Answer:
"""
prompt_template = ChatPromptTemplate.from_template(TEMPLATE)

def retrieve(query: str):
    """
    Retrieve relevant documents for a query.
    
    This function uses a hybrid approach:
    1. If section numbers are explicitly mentioned, retrieve those sections directly
    2. Otherwise, perform a semantic search using the query
    
    Args:
        query (str): The user's query
        
    Returns:
        list: Retrieved documents
    """
    # Extract section numbers from the query
    secs = extract_section_numbers(query)
    
    if secs:
        # If section numbers are found, filter by those sections
        flt = Filter(
            must=[FieldCondition(key="metadata.sections", match=MatchAny(any=secs))]
        )
        docs = vectorstore.similarity_search(query="", filter=flt)
    else:
        # Otherwise, perform semantic search
        docs = vectorstore.similarity_search(query, k=5)
    
    return docs

def docs_to_text(docs):
    """
    Extract page content from document objects.
    
    Args:
        docs (list): List of document objects
        
    Returns:
        list[str]: List of document contents
    """
    return [d.page_content for d in docs]


# Process each query in the test dataset
print("Processing test queries and generating responses...")
dataset = []
for _, row in df.iterrows():
    # Extract query and reference data
    query = row['user_input']
    ref_ctxs = row['reference_contexts']   # Reference contexts from the test dataset
    reference = row['reference']           # Reference answer

    # Retrieve relevant documents
    docs = retrieve(query)
    retrieved_ctxs = docs_to_text(docs)

    # Generate response using the RAG pipeline
    resp = (prompt_template | llm).invoke({
        "context": "\n\n".join(retrieved_ctxs),
        "question": query
    })

    # Store results
    dataset.append({
        "user_input":         query,
        "reference_contexts": ref_ctxs,
        "retrieved_contexts": retrieved_ctxs,
        "response":           resp.content,
        "reference":          reference
    })

# Prepare evaluation dataset (using first 30 samples)
print("Preparing evaluation dataset...")
evaluated_dataset = []

for i in range(0, min(30, len(dataset))):
    evaluated_dataset.append({
        "user_input": dataset[i]['user_input'],
        "retrieved_contexts": dataset[i]['retrieved_contexts'],
        "response": dataset[i]['response'],
        "reference": dataset[i]['reference']
    })

# Convert to RAGAS evaluation dataset format
evaluation_dataset = EvaluationDataset.from_list(evaluated_dataset)

# Run evaluation with RAGAS metrics
print("Running evaluation with RAGAS metrics...")
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        LLMContextRecall(),    # Measures how well the system retrieves relevant contexts
        Faithfulness(),        # Measures how faithful the response is to the retrieved contexts
        FactualCorrectness()   # Measures the factual accuracy of the response
    ],
    llm=evaluator_llm
)

# Print evaluation results
print("\nEvaluation Results:")
print(result)