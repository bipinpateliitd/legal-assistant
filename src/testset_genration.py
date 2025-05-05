from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-nano"))
from dotenv import load_dotenv
import pandas as pd
import os
from ragas.testset import TestsetGenerator
load_dotenv()

embedding_model = "text-embedding-3-large"

embedding = OpenAIEmbeddings(model=embedding_model)
generator_embeddings = LangchainEmbeddingsWrapper(embedding)

# Define path to sections directory
path = "/home/bipin/Documents/projects/legal_assistant/data/sections"

# Create loader for .txt files

loader = DirectoryLoader(
    path,
    glob="**/*.txt",  # This will match all .txt files in the directory
    show_progress=True  # Shows a progress bar while loading
)
# loader = TextLoader("/home/bipin/Documents/projects/legal_assistant/data/sections/bns_section_2.txt")

# Load all documents
docs = loader.load()



generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=30)

dataset.to_pandas()
dataset.to_csv("/home/bipin/Documents/projects/legal_assistant/data/bns_testset.csv")


