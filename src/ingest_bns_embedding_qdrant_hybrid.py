from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
import pandas as pd
from dotenv import load_dotenv
import os
import json


load_dotenv()

# Configuration
sections = 358
chunk_size = 900
chunk_overlap = 180
embedding_model = "text-embedding-3-large"
collection_name = "bns_sections_hybrid"
dense_embedding = OpenAIEmbeddings(model=embedding_model)
sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")

# Qdrant configuration
url = os.getenv("QDRANT_CLOUD_URL")
api_key = os.getenv("QDRANT_API_KEY")


client = QdrantClient(
    url=url,
    api_key=api_key,
    prefer_grpc=True,
    timeout= 60.0,
)


client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=3072, distance=Distance.COSINE)  # OpenAI embedding size
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
    )

# Load metadata
sections_meta_path = "/home/bipin/Documents/projects/legal_assistant/data/metadata/bns_sections_metadata.csv"
keywords_meta_path = "/home/bipin/Documents/projects/legal_assistant/data/metadata/bns_top_keywords.csv"


df_keywords = pd.read_csv(keywords_meta_path)
df_sections = pd.read_csv(sections_meta_path)


# Data loading and metadata adding
bns_documents = []
for i in range(1, sections + 1):
    filepath = f"/home/bipin/Documents/projects/legal_assistant/data/sections/bns_section_{i}.txt"

        
    loader = TextLoader(filepath)
    meta_section = df_sections[df_sections["section"] == i]
    meta_keyword = df_keywords[df_keywords["section"] == i]

        
    docs = loader.load()
    docs[0].metadata["sections"] = int(meta_section["section"].values[0])
    docs[0].metadata["title"] = meta_section["title"].values[0]
    docs[0].metadata["description"] = meta_section["description"].values[0]
    keywords_list = json.loads(meta_keyword["top_keywords"].values[0])
    docs[0].metadata["keywords"] = keywords_list  # Convert list to string
    bns_documents.append(docs[0])
        

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
chunks = text_splitter.split_documents(bns_documents)



# Embedding calculation and saving in vector dB

qdrant = QdrantVectorStore(
    client=client,
    collection_name= collection_name ,
    embedding=dense_embedding,
    sparse_embedding=sparse_embedding,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

qdrant.add_documents(
    documents=chunks,
    batch_size=64
    )