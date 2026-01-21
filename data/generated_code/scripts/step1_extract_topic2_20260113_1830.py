import PyPDF2
import json
import re
from collections import defaultdict
import os

# Ensure the output directory exists
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def identify_topic_2_sections(text):
    """Identify sections related to 'Topic 2' in the text."""
    # Look for patterns like "Topic 2", "Chapter 2", "Section 2", etc.
    topic_pattern = r'(?:Topic|Chapter|Section)\s*2(?:\s*[:\-]?\s*[A-Za-z0-9\s]*)?'
    
    # Find all matches
    matches = re.finditer(topic_pattern, text, re.IGNORECASE)
    
    # Extract context around each match
    sections = []
    for match in matches:
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end]
        sections.append({
            'match': match.group(),
            'context': context,
            'position': match.start()
        })
    
    return sections

def parse_outline(text):
    """Parse text into structured outline components."""
    outline = {
        'headings': [],
        'subheadings': [],
        'key_definitions': [],
        'formulas': [],
        'code_snippets': [],
        'figures_tables': []
    }
    
    # Pattern matching for different elements
    # Headings (typically bold or large font)
    heading_pattern = r'^\s*(?:[0-9]+\.[0-9]+\.\s*)?[A-Z][^.!?]*[.!?]' 
    headings = re.findall(heading_pattern, text, re.MULTILINE)
    outline['headings'] = [h.strip() for h in headings if len(h.strip()) > 10]
    
    # Subheadings (usually smaller numbered sections)
    subheading_pattern = r'^\s*[0-9]+\.[0-9]+\s+[A-Z][^.!?]*[.!?]'
    subheadings = re.findall(subheading_pattern, text, re.MULTILINE)
    outline['subheadings'] = [s.strip() for s in subheadings if len(s.strip()) > 10]
    
    # Key definitions (look for terms followed by definitions)
    definition_pattern = r'\b([A-Z][a-zA-Z\s]+?)\s*[:\-]\s*(.*?)(?=\n\n|\Z)'
    definitions = re.findall(definition_pattern, text, re.DOTALL)
    outline['key_definitions'] = [{'term': d[0].strip(), 'definition': d[1].strip()} for d in definitions if len(d[1].strip()) > 20]
    
    # Formulas (look for mathematical expressions)
    formula_pattern = r'(?:\$[^$]*\$|\\begin\{equation\}.*?\\end\{equation\})'
    formulas = re.findall(formula_pattern, text, re.DOTALL)
    outline['formulas'] = [f.strip() for f in formulas if len(f.strip()) > 5]
    
    # Code snippets (look for indented code or fenced code blocks)
    code_pattern = r'