from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
sections = 358
chunk_size = 900
chunk_overlap = 180
embedding_model = "text-embedding-3-large"
persist_directory = "/home/bipin/Documents/projects/legal_assistant/db/bns_sections_chroma"

# Validate configuration
if chunk_overlap >= chunk_size:
    raise ValueError("chunk_overlap must be smaller than chunk_size")

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Load metadata
sections_meta_path = "/home/bipin/Documents/projects/legal_assistant/data/metadata/bns_sections_metadata.csv"
keywords_meta_path = "/home/bipin/Documents/projects/legal_assistant/data/metadata/bns_top_keywords.csv"

try:
    df_keywords = pd.read_csv(keywords_meta_path)
    df_sections = pd.read_csv(sections_meta_path)
except Exception as e:
    logger.error(f"Error loading metadata files: {str(e)}")
    raise

# Data loading and metadata adding
documents = []
for i in tqdm(range(1, sections + 1), desc="Loading documents"):
    try:
        filepath = f"/home/bipin/Documents/projects/legal_assistant/data/sections/bns_section_{i}.txt"
        
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            continue
            
        loader = TextLoader(filepath)
        meta_section = df_sections[df_sections["section"] == i]
        meta_keyword = df_keywords[df_keywords["section"] == i]
        
        # Verify metadata exists
        if meta_section.empty or meta_keyword.empty:
            logger.warning(f"Missing metadata for section {i}, skipping...")
            continue
            
        docs = loader.load()
        docs[0].metadata["sections"] = int(meta_section["section"].values[0])
        docs[0].metadata["title"] = meta_section["title"].values[0]
        docs[0].metadata["description"] = meta_section["description"].values[0]
        keywords_list = json.loads(meta_keyword["top_keywords"].values[0])
        docs[0].metadata["keywords"] = ", ".join(keywords_list)
        documents.append(docs[0])
        
    except Exception as e:
        logger.error(f"Error processing section {i}: {str(e)}")

# Validate documents were loaded
if not documents:
    raise ValueError("No documents were loaded successfully")

logger.info(f"Successfully loaded {len(documents)} documents")

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
chunks = text_splitter.split_documents(documents)

if not chunks:
    raise ValueError("No chunks were created from documents")

logger.info(f"Created {len(chunks)} chunks from documents")

# Embedding calculation
try:
    embedding = OpenAIEmbeddings(model=embedding_model)
except Exception as e:
    logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
    raise

# Saving embedding in Chroma vector_db
try:
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name="bns_sections",
    )
    logger.info(f"Successfully saved embeddings to {persist_directory}")
except Exception as e:
    logger.error(f"Error saving to vector database: {str(e)}")
    raise