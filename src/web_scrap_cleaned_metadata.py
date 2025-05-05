#!/usr/bin/env python3
"""
Bhartiya Nyaya Sanhita (BNS) Metadata Scraper

This script scrapes metadata (section number, title, and description) for all sections of the 
Bhartiya Nyaya Sanhita from devgan.in. It uses multithreading to efficiently process multiple 
sections in parallel and saves the collected metadata to a CSV file.

The metadata includes:
- Section number
- Section title
- Section description

This information is useful for creating searchable indices, generating summaries, or enhancing
the user interface of legal information retrieval systems.
"""

import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import WebBaseLoader
from data_cleaning import clean_bns_metadata

# Constants
sections = 358  # Total number of sections in the Bhartiya Nyaya Sanhita
max_threads = 10  # Number of parallel threads for scraping

# Path where the metadata CSV will be saved
output_csv = "/home/bipin/Documents/projects/legal_assistant/data/metadata/bns_sections_metadata.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Initialize empty DataFrame to store metadata
df = pd.DataFrame(columns=["section", "title", "description"])

def process_section(section):
    """
    Process a single BNS section: scrape and clean its metadata.
    
    Args:
        section (int): The section number to process
        
    Returns:
        dict: A dictionary containing the section number, title, and description,
              or None if an error occurs
        
    Raises:
        Various exceptions may be raised during web scraping or metadata processing
    """
    try:
        # Construct URL for the specific section
        url = f"https://devgan.in/bns/section/{section}/"
        
        # Load the web page
        loader = WebBaseLoader(url)
        data = loader.load()
        
        # Extract and clean the metadata
        raw_metadata = data[0].metadata
        cleaned = clean_bns_metadata(raw_metadata, section)
        
        return {
            "section": cleaned["section"],
            "title": cleaned["title"],
            "description": cleaned["description"]
        }
    except Exception as e:
        print(f"Error processing section {section}: {str(e)}")
        return None

# Main execution: Run metadata scraping in parallel using ThreadPoolExecutor
print(f"Starting to scrape metadata for {sections} BNS sections using {max_threads} threads...")
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    # Submit all section processing tasks to the executor
    futures = {executor.submit(process_section, sec): sec for sec in range(1, sections + 1)}
    
    # Process results as they complete
    for future in as_completed(futures):
        result = future.result()
        if result:
            # Add the result to the DataFrame
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
            print(f"Processed section {result['section']}")

# Save the collected metadata to CSV
df.to_csv(output_csv, index=False)
print(f"Metadata scraping complete. CSV saved at: {output_csv}")
