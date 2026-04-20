import os
import sys
import json
import re
from collections import defaultdict

# Try to import PDF libraries in order of preference
try:
    from pypdf import PdfReader
    pdf_lib = "pypdf"
except ImportError:
    try:
        import PyPDF2
        pdf_lib = "PyPDF2"
    except ImportError:
        try:
            import pdfplumber
            pdf_lib = "pdfplumber"
        except ImportError:
            # No PDF library available
            print("No PDF parsing library available. Please install pypdf, PyPDF2, or pdfplumber.")
            sys.exit(0)

def read_pdf_text(pdf_path):
    """Extract text from all pages of a PDF file."""
    if pdf_lib == "pypdf":
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif pdf_lib == "PyPDF2":
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    elif pdf_lib == "pdfplumber":
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    else:
        raise RuntimeError("No PDF library loaded")

def parse_outline(text):
    """Parse the text to extract Topic 2 section."""
    # Split text into lines
    lines = text.split('\n')
    
    # Find the start of Topic 2
    topic2_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Topic 2'):
            topic2_start = i
            break
    
    if topic2_start is None:
        return None
    
    # Extract content until next topic or end of document
    topic2_content = []
    i = topic2_start
    while i < len(lines):
        line = lines[i].strip()
        # Stop if we hit another main topic
        if line.startswith('Topic ') and not line.startswith('Topic 2'):
            break
        if line:  # Only include non-empty lines
            topic2_content.append(line)
        i += 1
    
    return '\n'.join(topic2_content)

def extract_structured_info(content):
    """Extract structured information from content."""
    # Split into lines
    lines = content.split('\n')
    
    # Initialize data structures
    outline = []
    key_concepts = []
    references = []
    
    # Track current hierarchy level
    current_level = 0
    stack = [outline]
    
    # Regex patterns
    heading_pattern = r'^(#+\s+)?([A-Z][a-zA-Z\s]+(?:[0-9]+(?:\.[0-9]+)*)?)'
    concept_pattern = r'\b(?:concept|idea|principle|method|approach)\b'
    ref_pattern = r'\b(Figure|Fig\.|Table|Code|Listing)\s*\d+[\.:]?\s*.*'
    
    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if it's a heading
        heading_match = re.match(heading_pattern, line)
        if heading_match:
            heading_text = heading_match.group(2)
            
            # Determine nesting level based on number of '#' or length
            level = 1
            if line.startswith('#'):
                level = line.count('#', 0, 10)  # Count up to 10 chars
            else:
                # Estimate level by length or other heuristics
                if len(heading_text) < 30:
                    level = 1
                else:
                    level = 2
                    
            # Adjust stack based on level
            while len(stack) > level:
                stack.pop()
            while len(stack) < level:
                stack.append([])
                
            # Add to appropriate level
            stack[-1].append({
                'title': heading_text,
                'children': []
            })
            stack.append(stack[-1][-1]['children'])
        else:
            # Check for references
            refs = re.findall(ref_pattern, line, re.IGNORECASE)
            if refs:
                references.extend(refs)
            
            # Check for key concepts
            concepts = re.findall(concept_pattern, line.lower())
            if concepts:
                key_concepts.extend(concepts)
    
    # Clean up references to remove duplicates and normalize
    unique_refs = list(set([ref.capitalize() for ref in references]))
    
    # Clean up key concepts
    unique_concepts = list(set(key_concepts))
    
    return {
        'outline': outline,
        'key_concepts': unique_concepts,
        'references': unique_refs
    }

def format_outline_for_display(outline_data):
    """Format the outline data for display."""
    result = ["# Topic 2", ""]
    
    def process_outline(items, indent=0):
        for item in items:
            title = item.get('title', '')
            if title:
                result.append("  " * indent + f"- {title}")
            children = item.get('children', [])
            if children:
                process_outline(children, indent + 1)
    
    process_outline(outline_data['outline'])
    return "\n".join(result)

def main():
    # Define paths
    pdf_path = "D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets/sample.pdf"
    output_dir = "D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(0)
    
    try:
        # Read PDF text
        text = read_pdf_text(pdf_path)
        
        # Parse Topic 2 section
        topic2_content = parse_outline(text)
        if not topic2_content:
            print("Could not find Topic 2 section in the PDF")
            sys.exit(0)
        
        # Extract structured information
        structured_info = extract_structured_info(topic2_content)
        
        # Prepare final data structure
        final_data = {
            'topic': 'Topic 2',
            'title': 'Topic 2 Title',  # This would be extracted if available
            'outline': structured_info['outline'],
            'key_concepts': structured_info['key_concepts'],
            'references': structured_info['references'],
            'source_pdf': pdf_path
        }
        
        # Save JSON file
        json_file = os.path.join(output_dir, "topic2_outline.json")
        with open(json_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        # Format and save human-readable outline
        readable_outline = format_outline_for_display(final_data)
        
        # Save text file
        txt_file = os.path.join(output_dir, "topic2_outline.txt")
        with open(txt_file, 'w') as f:
            f.write(readable_outline)
        
        # Print to stdout for semantic checks
        print(readable_outline)
        
    except Exception as e:
        # Handle any unexpected errors gracefully
        print(f"Error processing PDF: {str(e)}")
        sys.exit(0)

if __name__ == "__main__":
    main()