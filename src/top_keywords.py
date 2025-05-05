#!/usr/bin/env python3
"""
Bhartiya Nyaya Sanhita (BNS) Keyword Extraction

This script extracts the most relevant legal keywords from each section of the Bhartiya Nyaya Sanhita.
It uses a two-step approach:
1. Initial keyword extraction using spaCy NLP to identify potential keywords
2. Refinement using a language model to select the most legally relevant keywords

The final output is a CSV file containing the top 20 keywords for each section, which can be used
for search enhancement, topic modeling, or other text analysis purposes.
"""

import os
import pandas as pd
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List
import spacy
from collections import Counter
from langchain_openai import ChatOpenAI

# Load environment variables
from dotenv import load_dotenv
path ="/home/bipin/Documents/projects/legal_assistant/.env"
load_dotenv(dotenv_path=path)

# Constants and paths
TEXT_DIR = "/home/bipin/Documents/projects/legal_gpt/data/sections"  # Directory containing section text files
CSV_PATH = "/home/bipin/Documents/projects/legal_assistant/data/metadata/bns_top_keywords.csv"  # Output CSV path

# Ensure output directory exists
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# Define structured output model for LLM
class Top20Keywords(BaseModel):
    """Structured output model for top 20 unique legally relevant keywords"""
    top_keywords: List[str] = Field(..., description="Exactly 20 unique top legally relevant keywords")

# Load spaCy NLP model for initial keyword extraction
nlp = spacy.load("en_core_web_sm")

def extract_keywords_spacy(text: str, top_n: int = 50) -> List[str]:
    """
    Extract potential keywords from text using spaCy NLP.
    
    Args:
        text (str): The text to extract keywords from
        top_n (int): Number of top keywords to return
        
    Returns:
        List[str]: List of the top_n most frequent nouns in the text
    """
    # Process the text with spaCy
    doc = nlp(text.lower())
    
    # Extract words that meet our criteria for potential keywords:
    # - Not a stop word (common words like 'the', 'is', etc.)
    # - Not punctuation
    # - Longer than 3 characters
    # - Is a noun
    words = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 3 and token.pos_ == "NOUN"
    ]
    
    # Count word frequencies
    freqs = Counter(words)
    
    # Return the most common words
    return [w for w, _ in freqs.most_common(top_n)]


# Initialize language model for keyword refinement
llm = ChatOpenAI(model="gpt-4.1-nano", max_tokens=200, temperature=0.1)
structured_llm = llm.with_structured_output(Top20Keywords)

# Storage for results
records = []

print(f"Starting keyword extraction for BNS sections...")

# Process each section
for section in range(1, 359):
    # Construct path to the section text file
    txt_path = os.path.join(TEXT_DIR, f"bns_section_{section}.txt")
    
    # Skip if file doesn't exist
    if not os.path.exists(txt_path):
        print(f"Section {section} file not found, skipping.")
        continue

    # Read the section text
    with open(txt_path, encoding="utf-8") as f:
        text = f.read()

    # Step 1: Extract candidate keywords using spaCy
    candidates = extract_keywords_spacy(text, top_n=50)

    # Step 2: Use LLM to refine and select the most legally relevant keywords
    prompt = f"""
I have these candidate keywords from section {section} of the Bhartiya Nyaya Sanhita:

{', '.join(candidates)}

Please return only the top 20 unique keywords that are topically and legally relevant to criminal or procedural law in India.
""".strip()

    # Get structured response from LLM
    resp = structured_llm.invoke(prompt)
    data = resp.model_dump()
    
    # Ensure we have exactly 20 unique keywords
    unique_keywords = list(dict.fromkeys(data["top_keywords"]))[:20]

    # Add to our records
    records.append({
        "section": section,
        "top_keywords": ", ".join(unique_keywords)
    })
    print(f"✓ Section {section} processed with keywords: {unique_keywords[:3]}...")

# Create DataFrame and sort by section number
df = pd.DataFrame(records).sort_values("section")

# Save to CSV
df.to_csv(CSV_PATH, index=False)
print(f"Keyword extraction complete. Saved CSV → {CSV_PATH}")
