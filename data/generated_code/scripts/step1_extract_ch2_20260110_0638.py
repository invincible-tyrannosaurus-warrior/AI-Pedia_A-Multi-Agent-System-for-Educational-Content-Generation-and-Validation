import os
import json
import re
import fitz  # PyMuPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Define paths
pdf_path = "D:/L3/Individual_project/AI_Pedia_Local/data/uploaded_files/Chapter_2.pdf"
output_dir = "D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets"

# Initialize data structure
chapter_data = {
    "title": "Chapter 2",
    "outline": [],
    "equations": [],
    "figures_tables": [],
    "code_examples": []
}

# Helper function to extract text from a page
def extract_text_from_page(page):
    return page.get_text()

# Helper function to find equations (simple pattern matching)
def find_equations(text):
    # Pattern to match LaTeX-style equations (both inline and display)
    equation_pattern = r'\$[^$]*\$|\\begin\{equation\}.*?\\end\{equation\}'
    equations = re.findall(equation_pattern, text, re.DOTALL)
    return equations

# Helper function to find figures/tables references
def find_figures_tables(text):
    # Pattern to match figure/table references
    fig_table_pattern = r'(Figure|Table)\s+\d+\.\d+'
    references = re.findall(fig_table_pattern, text)
    return references

# Helper function to find code examples
def find_code_examples(text):
    # Pattern to match code blocks (Python code typically between triple backticks)
    code_pattern = r'