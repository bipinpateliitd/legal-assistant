import re

def clean_bns_page_content(text: str) -> str:
    # Remove JavaScript and navigation text
    text = re.sub(r'No Javascript.*?Reload this page!', '', text, flags=re.DOTALL)
    text = re.sub(r'(Top|Prev|Index|Next|Messages|All Sections Lists:).*$', '', text, flags=re.MULTILINE)

    # Remove devgan branding and titles
    text = re.sub(r'BNS Section \d+.*?Devgan\.in', '', text)
    text = re.sub(r'Devgan\.in BNS', '', text, flags=re.IGNORECASE)

    # Remove update info, footer, and credit line
    text = re.sub(r'©.*?A Lawyers Reference™.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'Updated: .*?\d{4}', '', text)
    text = re.sub(r'By Raman Devgan', '', text, flags=re.IGNORECASE)

    # Replace unwanted characters and clean up whitespace
    text = text.replace('\xa0', ' ').replace('\n', ' ').strip()
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'By Raman Devgan.*?All Sections Lists:.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'Bharatiya Nyaya Sanhita Home', '', text)
    text= re.sub(r'\n{3,}', '\n\n', text)

    # Normalize tabs: turn tabs into 4 spaces
    text = text.replace('\t', '    ')

    # Remove extra spaces at line beginnings/ends
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)

    # Ensure consistent line endings
    text = re.sub(r'[ ]{2,}', ' ', text)  # Reduce multiple spaces to 1
    text = text.strip()

    return text





def clean_bns_metadata(metadata: dict, section_number :int) -> dict:
    cleaned_metadata = metadata.copy()
    if 'title' in cleaned_metadata:
        cleaned_metadata['title'] = re.sub(r'\s*\|\s*Devgan\.in\s*$', '', cleaned_metadata['title']).strip()
    if 'description' in cleaned_metadata:
        cleaned_metadata['description'] = re.sub(
            r'from the Bharatiya Nyaya Sanhita, by Advocate Raman Devgan',
            '',
            cleaned_metadata['description'],
            flags=re.IGNORECASE
        ).strip()
        cleaned_metadata['description'] = re.sub(r',\s*$', '', cleaned_metadata['description'])
        cleaned_metadata["section"] = section_number
    return cleaned_metadata




