#!/usr/bin/env python3
"""
Bhartiya Nyaya Sanhita (BNS) Section Data Scraper

This script scrapes the content of all sections of the Bhartiya Nyaya Sanhita from devgan.in,
cleans the text data, and saves each section as a separate text file. The script uses
multithreading to efficiently scrape multiple sections in parallel.

The cleaned text files are saved in the specified output directory and can be used for
further processing, analysis, or as part of a legal information retrieval system.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import WebBaseLoader
from data_cleaning import clean_bns_page_content, clean_bns_metadata

# Constants
sections = 358  # Total number of sections in the Bhartiya Nyaya Sanhita
max_threads = 10  # Number of parallel threads for scraping (adjust based on system capacity)

# Directory where cleaned section text files will be saved
output_dir ="/home/bipin/Documents/projects/legal_assistant/data/sections"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


def process_section(section):
    """
    Process a single BNS section: scrape, clean, and save to a text file.
    
    Args:
        section (int): The section number to process
        
    Returns:
        str: A message indicating the section was successfully saved
        
    Raises:
        Various exceptions may be raised during web scraping or file operations
    """
    try:
        # Construct URL for the specific section
        url = f"https://devgan.in/bns/section/{section}/"
        
        # Load the web page content
        loader = WebBaseLoader(url)
        data = loader.load()
        
        # Clean the raw page content using the imported cleaning function
        cleaned_text = clean_bns_page_content(data[0].page_content)

        # Save the cleaned text to a file
        file_path = os.path.join(output_dir, f"bns_section_{section}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        return f"Section {section} saved."
    except Exception as e:
        return f"Error processing section {section}: {str(e)}"

# Main execution: Run scraping in parallel using ThreadPoolExecutor
print(f"Starting to scrape {sections} BNS sections using {max_threads} threads...")
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    # Submit all section processing tasks to the executor
    futures = {executor.submit(process_section, sec): sec for sec in range(1, sections + 1)}

    # Process results as they complete
    for future in as_completed(futures):
        print(future.result())

print(f"Scraping complete. Files saved to {output_dir}")
