#!/usr/bin/env python3
"""
Content-Based Markdown Image Generator

This script enhances markdown files by adding images generated from the actual content of sections,
rather than just using headings. It analyzes the text content to create meaningful image prompts
and inserts images at appropriate locations.

Usage:
    python content_based_image_generator.py --input input_folder --output output_folder [--max_images MAX_IMAGES] [--table_images]
"""

import os
import re
import urllib.parse
from pathlib import Path
import argparse
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
import random
import requests  # Add this import at the top
from datetime import datetime, timedelta
import uuid

# Download NLTK resources (expanded to include all needed resources)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# This function is a workaround for the punkt_tab issue
def safe_sent_tokenize(text):
    """
    A safer version of sent_tokenize that falls back to simple regex-based splitting
    if the NLTK tokenizer fails.
    """
    try:
        return sent_tokenize(text)
    except LookupError:
        # Simple fallback tokenizer using regular expressions
        # Split on periods followed by space and capital letter, 
        # or question/exclamation marks, or new lines
        return re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*[\n\r]+', text)

def extract_meaningful_phrases(text, num_phrases=3, words_per_phrase=3):
    """
    Extract meaningful phrases from text that can be used for image generation.
    Returns a list of phrases that capture the essence of the content.
    """
    # Remove markdown syntax
    text = re.sub(r'#+\s+', '', text)  # Remove headings
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove links
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove images
    text = re.sub(r'`.*?`', '', text)  # Remove inline code
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
    
    # Tokenize into sentences using our safe tokenizer
    sentences = safe_sent_tokenize(text)
    
    # Identify important sentences (first sentence often contains key info)
    important_sentences = sentences[:min(5, len(sentences))]
    
    # Also include sentences with important keywords
    keywords = ["key", "important", "essential", "fundamental", "main", "critical", "vital", 
                "example", "specifically", "notably", "primarily", "central", "core"]
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in keywords) and sentence not in important_sentences:
            important_sentences.append(sentence)
    
    # Extract nouns and adjectives - they're most important for visual representation
    # Since we don't have POS tagging here, we'll use a heuristic approach
    stop_words = set(stopwords.words('english'))
    all_words = []
    
    for sentence in important_sentences:
        words = sentence.lower().split()
        words = [word.strip(string.punctuation) for word in words if word.strip(string.punctuation)]
        words = [word for word in words if word not in stop_words and len(word) > 2]
        all_words.extend(words)
    
    # Count word frequency
    word_counter = Counter(all_words)
    
    # Find the most common words
    common_words = [word for word, _ in word_counter.most_common(15)]
    
    # Create phrases by combining words
    phrases = []
    if len(common_words) >= words_per_phrase:
        # Create deterministic phrases from most common words
        for i in range(min(num_phrases, len(common_words) // words_per_phrase)):
            if i * words_per_phrase + words_per_phrase <= len(common_words):
                phrase = " ".join(common_words[i * words_per_phrase:(i * words_per_phrase) + words_per_phrase])
                phrases.append(phrase)
    
    # If we couldn't create enough phrases, add some random combinations
    while len(phrases) < num_phrases and len(common_words) >= 2:
        sample_size = min(words_per_phrase, len(common_words))
        phrase = " ".join(random.sample(common_words, sample_size))
        if phrase not in phrases:
            phrases.append(phrase)
    
    return phrases

def analyze_content_context(text):
    """
    Analyze the context of the content to determine what type of image would be appropriate.
    Returns a context descriptor (e.g., "abstract", "concept", "symbol").
    Added handling for sensitive topics.
    """
    # Define sensitive topic indicators
    sensitive_indicators = [
        "race", "ethnic", "gender", "sexuality", "religion", 
        "african", "asian", "european", "caucasian", "white", 
        "black", "hispanic", "latino", "indigenous", "native",
        "political", "controversial", "sensitive"
    ]
    
    # Check if content contains sensitive topics
    text_lower = text.lower()
    if any(indicator in text_lower for indicator in sensitive_indicators):
        return "abstract"
    
    # Enhanced context indicators with more specific categories
    context_indicators = {
        "portrait": ["person", "people", "face", "individual", "character", "figure", "portrait", 
                    "human", "man", "woman", "child", "children", "family", "group"],
        "landscape": ["scene", "nature", "environment", "vista", "panorama", "outdoor", "mountain", 
                    "forest", "beach", "ocean", "river", "sky", "weather", "field", "landscape", 
                    "garden", "park", "sunset", "sunrise"],
        "concept": ["idea", "concept", "abstract", "theory", "principle", "approach", "method", 
                   "philosophy", "strategy", "paradigm", "framework", "model", "notion"],
        "diagram": ["process", "flow", "structure", "system", "relationship", "framework", "chart", 
                   "graph", "data", "steps", "cycle", "sequence", "hierarchy", "method", "technique"],
        "object": ["item", "product", "device", "tool", "object", "instrument", "machine", "equipment", 
                  "gadget", "apparatus", "artifact", "material", "component", "container"],
        "architecture": ["building", "structure", "architecture", "house", "office", "skyscraper", 
                        "bridge", "monument", "construction", "design", "interior", "exterior"],
        "technology": ["computer", "software", "hardware", "algorithm", "program", "code", "application", 
                      "digital", "tech", "technological", "electronic", "device", "network", "internet"],
        "food": ["food", "meal", "dish", "cuisine", "recipe", "ingredient", "vegetable", "fruit", 
                "meat", "dessert", "beverage", "drink", "cooking", "baking", "restaurant"],
        "medical": ["medical", "health", "disease", "treatment", "therapy", "medicine", "hospital", 
                   "doctor", "patient", "anatomy", "physiology", "diagnosis", "symptom"],
        "symbol": ["symbol", "icon", "representation", "emblem", "sign", "logo", "metaphor", 
                  "allegory", "significance", "meaning", "symbolic"]
    }
    
    # Count occurrences of context indicators
    context_counts = {context: 0 for context in context_indicators}
    
    for context, indicators in context_indicators.items():
        for indicator in indicators:
            # Add extra weight for exact matches to improve accuracy
            if indicator in text_lower.split():
                context_counts[context] += 2
            else:
                context_counts[context] += text_lower.count(indicator)
    
    # Determine the most likely context
    if max(context_counts.values()) > 0:
        primary_context = max(context_counts.items(), key=lambda x: x[1])[0]
    else:
        # Default to concept if no strong indicators are found
        primary_context = "concept"
    
    return primary_context

def generate_image_style(content):
    """
    Determine an appropriate image style based on the content.
    Returns a style descriptor (e.g., "photorealistic", "illustration", "artistic").
    """
    # Enhanced style indicators with more specific terms
    style_indicators = {
        "photorealistic": ["realistic", "photo", "real", "photograph", "actual", "natural", "lifelike",
                          "detailed", "high-definition", "true-to-life", "photography", "camera", 
                          "documentary", "authentic", "genuine"],
        "illustration": ["illustration", "drawing", "cartoon", "sketch", "icon", "graphic",
                        "caricature", "simplified", "outlined", "drawn", "stylized drawing", 
                        "hand-drawn", "line art", "diagrammatic"],
        "artistic": ["art", "artistic", "creative", "abstract", "stylized", "expressive", "painting",
                    "modernist", "contemporary", "impressionist", "vibrant", "colorful", "visual", 
                    "aesthetic", "beautiful", "striking"],
        "technical": ["technical", "diagram", "schematic", "blueprint", "detailed", "engineering",
                     "analytical", "instructional", "educational", "explanatory", "informative", 
                     "instructive", "research", "professional", "academic"]
    }
    
    # Count occurrences of style indicators
    style_counts = {style: 0 for style in style_indicators}
    content_lower = content.lower()
    
    for style, indicators in style_indicators.items():
        for indicator in indicators:
            # Add extra weight for exact matches to improve accuracy
            if indicator in content_lower.split():
                style_counts[style] += 2
            else:
                style_counts[style] += content_lower.count(indicator)
    
    # Determine the most likely style
    if max(style_counts.values()) > 0:
        primary_style = max(style_counts.items(), key=lambda x: x[1])[0]
    else:
        # Default to photorealistic if no strong indicators are found
        primary_style = "photorealistic"
    
    return primary_style

def generate_advanced_image_prompt(section_content, heading_text=None):
    """
    Generate an appropriate image prompt based on detailed content analysis.
    Creates more specific, detailed prompts that better reflect the content.
    """
    import re
    
    # Check for sensitive content first
    sensitive_indicators = [
        "race", "ethnic", "gender", "sexuality", "religion", 
        "african", "asian", "european", "caucasian", "white", 
        "black", "hispanic", "latino", "indigenous", "native",
        "political", "controversial", "sensitive"
    ]
    
    is_sensitive = any(indicator in section_content.lower() for indicator in sensitive_indicators)
    
    # Extract meaningful phrases from the content
    phrases = extract_meaningful_phrases(section_content)
    
    # Determine the context of the content
    context = analyze_content_context(section_content)
    
    # Determine appropriate image style
    style = generate_image_style(section_content)
    
    # For sensitive topics, use abstract representations
    if is_sensitive:
        # Create a more abstract prompt
        if heading_text:
            safe_heading = re.sub(r'\*\*|\*|__|\^|~|`', '', heading_text)
            base_prompt = f"abstract symbol representing {safe_heading}, geometric pattern"
        elif phrases:
            base_prompt = f"abstract representation of {', '.join(phrases)}, symbolic design"
        else:
            base_prompt = "abstract geometric pattern, symbolic representation"
            
        # Add abstract style modifiers
        style_modifier = "minimalist design, neutral colors, conceptual representation, no human faces, simple graphic"
        return f"{base_prompt}, {style_modifier}"
    
    # For non-sensitive content, create a more specific and descriptive prompt
    # Start with the heading or title if available
    if heading_text:
        clean_heading = re.sub(r'\*\*|\*|__|\^|~|`', '', heading_text)
        
        # Extract key nouns and adjectives from heading
        heading_words = clean_heading.split()
        # Focus on longer words (more likely to be meaningful)
        key_heading_words = [word for word in heading_words if len(word) > 3]
        
        # Create a more specific base prompt
        if key_heading_words:
            base_prompt = f"{clean_heading}, detailed visualization"
        else:
            base_prompt = f"detailed visualization of {clean_heading}"
            
        if phrases:
            # Add some of the key phrases for context - be very specific
            base_prompt += f" showing {', '.join(phrases[:2])}"
    elif phrases:
        # If no heading, use the extracted phrases as the main prompt
        # But be more specific about what we want to see
        base_prompt = f"{', '.join(phrases[:3])}, detailed visualization"
    else:
        base_prompt = "abstract concept visualization"
    
    # Extract any numerical data mentioned in the content to help with visualizing specifics
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', section_content)
    if numbers and len(numbers) >= 2:
        base_prompt += f" with data points {', '.join(numbers[:3])}"
    
    # Add style and quality modifiers - more specific ones for better results
    style_modifiers = {
        "photorealistic": "photorealistic rendering, high-quality, detailed 8k, realistic lighting, sharp focus",
        "illustration": "professional digital illustration, clean lines, vibrant colors, clear details, vector style",
        "artistic": "artistic rendering, creative style, expressive color palette, detailed texture, professional quality",
        "technical": "technical diagram, clear structure, precise lines, informational, labeled elements"
    }
    
    # Construct the final prompt with style
    prompt = f"{base_prompt}, {style_modifiers.get(style, 'high quality')}"
    
    # Add context-specific modifiers for more relevant images
    context_modifiers = {
        "portrait": "professional portrait style, clear subject focus, well composed",
        "landscape": "wide angle perspective, natural environment, atmospheric",
        "concept": "conceptual visualization, symbolic representation, clear meaning",
        "diagram": "organized layout, systematic presentation, labeled components",
        "object": "isolated object with details, neutral background, product photography",
        "architecture": "architectural visualization, structural details, professional perspective",
        "technology": "modern tech aesthetic, clean design, digital environment",
        "food": "appetizing presentation, professional food styling, clear details",
        "medical": "medical visualization, anatomically accurate, educational style",
        "symbol": "iconic representation, simplified but recognizable form"
    }
    
    prompt += f", {context_modifiers.get(context, '')}"
    
    # Add a final quality keyword and specifically request image to be related to content
    prompt += ", professional quality, content-specific visualization, high detail"
    
    seed = str(uuid.uuid4())[:8]
    prompt += f", seed {seed}"
    return prompt

def split_markdown_into_sections(markdown_content):
    """
    Split the markdown content into sections based on headings.
    Returns a list of tuples: (heading_match, section_content, heading_level)
    """
    # Find all headings
    headings = list(re.finditer(r'^(#+)\s+(.+)$', markdown_content, re.MULTILINE))
    
    sections = []
    for i, heading in enumerate(headings):
        # Get heading level
        heading_level = len(heading.group(1))
        
        # Get section content
        start = heading.end()
        end = headings[i+1].start() if i < len(headings) - 1 else len(markdown_content)
        section_content = markdown_content[start:end].strip()
        
        sections.append((heading, section_content, heading_level))
    
    return sections

def should_add_image_to_section(section_content, heading_level, min_length=200):
    """
    Determine if a section should have an image added based on:
    1. Length and richness of content
    2. Heading level (more likely for higher level headings)
    3. Content doesn't already have an image
    """
    # Skip if section already has an image
    if re.search(r'!\[.*?\]\(.*?\)', section_content):
        return False
    
    # More likely to add image to higher level headings
    importance_factor = 1.0 / heading_level if heading_level > 0 else 1.0
    
    # Minimum length requirement increases for lower level headings
    adjusted_min_length = min_length * (2 - importance_factor)
    
    # Check if section is long enough
    return len(section_content) >= adjusted_min_length

def insert_image_into_section(heading_match, section_content, image_filename, alt_text):
    """
    Insert an image link after the heading of a section using local image path.
    """
    return f"\n\n![{alt_text}](images/{image_filename})\n\n{section_content}"

def extract_tables(markdown_content):
    """
    Extract markdown tables from the content.
    Returns a list of tuples: (table_start_pos, table_end_pos, table_content)
    """
    tables = []
    
    # Regex pattern to find markdown tables
    # Look for lines starting with | and containing at least one |
    lines = markdown_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line looks like the start of a table
        if line.strip().startswith('|') and '|' in line[1:]:
            table_start = i
            table_lines = [line]
            
            # Look for the header separator line
            if i + 1 < len(lines) and re.match(r'\s*\|[\s\-:]*\|', lines[i + 1]):
                table_lines.append(lines[i + 1])
                i += 1
                
                # Continue collecting table rows
                while i + 1 < len(lines) and lines[i + 1].strip().startswith('|') and '|' in lines[i + 1][1:]:
                    i += 1
                    table_lines.append(lines[i])
                
                # Calculate positions in the original markdown
                start_pos = markdown_content.find(table_lines[0])
                end_pos = markdown_content.find(table_lines[-1]) + len(table_lines[-1])
                
                if start_pos != -1 and start_pos < end_pos:
                    table_content = '\n'.join(table_lines)
                    tables.append((start_pos, end_pos, table_content))
        
        i += 1
    
    return tables

def parse_table(table_content):
    """
    Parse markdown table into rows and cells.
    Returns header row and data rows.
    """
    lines = table_content.strip().split('\n')
    
    if len(lines) < 3:  # Need at least header, separator, and one data row
        return None, []
    
    # Parse header row
    header = [cell.strip() for cell in lines[0].split('|')[1:-1]]
    
    # Skip separator row (index 1)
    
    # Parse data rows
    data_rows = []
    for i in range(2, len(lines)):
        cells = [cell.strip() for cell in lines[i].split('|')[1:-1]]
        # Ensure we have the same number of cells as the header
        while len(cells) < len(header):
            cells.append('')
        data_rows.append(cells)
    
    return header, data_rows

def generate_table_row_image_prompt(row_content, table_headers=None):
    """
    Generate a specific image prompt based on table row content with improved handling
    for sensitive categories and abstract representations.
    
    Args:
        row_content (str): The content of the table row
        table_headers (list): Optional list of table headers for context
    
    Returns:
        str: An appropriate image prompt
    """
    # Clean the content by removing markdown formatting
    cleaned_content = re.sub(r'\*\*|\*|__|\^|~|`', '', row_content)
    
    # Check for sensitive categories like race, ethnicity, gender
    sensitive_terms = [
        "race", "ethnic", "gender", "sexuality", "religion", 
        "african", "asian", "european", "caucasian", "white", 
        "black", "hispanic", "latino", "indigenous", "native"
    ]
    
    is_sensitive = any(term in cleaned_content.lower() for term in sensitive_terms)
    
    # For sensitive categories, use abstract representations
    if is_sensitive:
        # Extract the primary category term
        words = cleaned_content.lower().split()
        # Remove punctuation
        words = [word.strip(string.punctuation) for word in words]
        
        # Find primary identifying term
        primary_term = None
        for term in words:
            if len(term) > 3:  # Skip short words
                primary_term = term
                break
        
        if primary_term:
            # Use abstract symbolism instead of human depictions
            return f"abstract symbol representing {primary_term}, geometric pattern, minimalist design, neutral colors, conceptual representation, no human faces, icon style, simple graphic"
        else:
            return "abstract geometric symbol, minimalist design, neutral colors, conceptual representation, icon style"
    
    # For non-sensitive content, extract meaningful words, removing common stopwords
    stop_words = set(stopwords.words('english'))
    words = cleaned_content.lower().split()
    words = [word.strip(string.punctuation) for word in words if word.strip(string.punctuation)]
    words = [word for word in words if word not in stop_words and len(word) > 2]
    # Get unique words only
    unique_words = list(dict.fromkeys(words))
    # Select the top 2-3 unique words for the prompt
    top_words = unique_words[:3]
    if not top_words:
        return "abstract concept, professional illustration, icon style, simple graphic"
    # Add a random seed to make images less repetitive
    seed = str(uuid.uuid4())[:8]
    prompt = f"{', '.join(top_words)}, {table_headers[0] if table_headers else ''}, detailed, seed {seed}"
    return prompt

def add_image_column_to_table(table_content, output_file, max_image_size=150):
    """
    Add an image column to a markdown table based on row content with improved handling.
    Fixed with proper URL handling and fallback image service.
    """
    import os
    import re
    import urllib.parse
    
    header, data_rows = parse_table(table_content)
    
    if header is None:
        return table_content
    
    sensitive_headers = ["race", "ethnicity", "gender", "religion", "nationality"]
    is_sensitive_table = any(any(term.lower() in h.lower() for term in sensitive_headers) for h in header)
    
    new_header = header + ["Image"]
    new_data_rows = []
    
    output_image_dir = os.path.join(os.path.dirname(output_file), "images")
    os.makedirs(output_image_dir, exist_ok=True)
    
    for idx, row in enumerate(data_rows):
        row_text = " ".join(row)
        # Ensure row length matches header length
        row_fixed = row[:len(header)] + [""] * (len(header) - len(row))
        enriched_row_text = " ".join([f"{header[i]}: {cell}" for i, cell in enumerate(row_fixed) if cell.strip()])
        
        image_prompt = generate_table_row_image_prompt(enriched_row_text, header if is_sensitive_table else None)
        encoded_prompt = urllib.parse.quote(image_prompt)
        
        # Fix URL construction
        if is_sensitive_table:
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={max_image_size}&height={max_image_size}&nologo=true&style=minimalist%20graphic"
        else:
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={max_image_size}&height={max_image_size}&nologo=true"
        
        image_filename = f"table_row_{idx+1}.png"
        full_image_path = os.path.join(output_image_dir, image_filename)
        
        print(f"Generating image for table row {idx+1}...")
        
        success = download_image(image_url, full_image_path)
        
        # If download fails, try a fallback image service
        if not success:
            # Use the first cell as fallback text
            fallback_text = row[0] if row and row[0] else f"Row {idx+1}"
            fallback_url = f"https://placehold.co/{max_image_size}x{max_image_size}/png?text={urllib.parse.quote(fallback_text[:15])}"
            success = download_image(fallback_url, full_image_path)
        
        if success:
            alt_text = row[0] if row and row[0] else "Table image"
            alt_text = re.sub(r'\*\*|\*|__|\^|~|`', '', alt_text)
            image_cell = f"![{alt_text}](images/{image_filename})"
            new_data_rows.append(row + [image_cell])
        else:
            # If all image attempts fail, just add a placeholder text
            new_data_rows.append(row + ["[Image unavailable]"])
    
    new_table = []
    new_table.append("| " + " | ".join(new_header) + " |")
    separators = ["---"] * len(new_header)
    new_table.append("| " + " | ".join(separators) + " |")
    for row in new_data_rows:
        new_table.append("| " + " | ".join(row) + " |")
    
    return "\n".join(new_table)

def process_tables(markdown_content, output_file, max_images=5):
    """
    Process all tables in the markdown content, adding image columns.
    Returns updated content and number of images added.
    """
    tables = extract_tables(markdown_content)
    updated_content = markdown_content
    images_added = 0
    for table_start, table_end, table_content in sorted(tables, reverse=True):
        if images_added >= max_images:
            break
        if "![" in table_content and "](" in table_content:
            continue
        modified_table = add_image_column_to_table(table_content, output_file)
        updated_content = (
            updated_content[:table_start] +
            modified_table +
            updated_content[table_end:]
        )
        header, data_rows = parse_table(table_content)
        if header is not None:
            images_added += len(data_rows)
    return updated_content, images_added

def process_paragraphs(markdown_content, output_file, max_images=2):
    """
    Identify significant paragraphs that would benefit from images.
    Adds images based on paragraph content rather than just headings.
    """
    updated_content = markdown_content
    images_added = 0

    # Split content into paragraphs by two or more newlines
    paragraphs = re.split(r'\n{2,}', markdown_content)
    # Keep track of positions for insertion
    positions = []
    pos = 0
    for para in paragraphs:
        start = updated_content.find(para, pos)
        if start != -1:
            positions.append((start, para))
            pos = start + len(para)

    # Sort paragraphs by length (longer paragraphs are likely more important)
    sorted_paragraphs = sorted(positions, key=lambda p: len(p[1]), reverse=True)

    for start_pos, paragraph_content in sorted_paragraphs:
        if images_added >= max_images:
            break
        # Skip short paragraphs or those that already have images, headings, lists, or code blocks
        if (len(paragraph_content.strip()) < 200 or
            paragraph_content.strip().startswith(('#', '*', '-', '`')) or
            paragraph_content.strip()[0:1].isdigit() or
            '![' in paragraph_content):
            continue

        # Generate image prompt from paragraph content - enhanced extraction
        # First extract the first sentence as it often contains the paragraph's main point
        first_sentence = safe_sent_tokenize(paragraph_content)[:1]
        # Add weight to this first sentence in our analysis
        weighted_content = ' '.join([s for s in first_sentence]) + ' ' + paragraph_content
        image_prompt = generate_advanced_image_prompt(weighted_content)
        encoded_prompt = urllib.parse.quote(image_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=200&height=200&nologo=true"
        
        # Get first few words for alt text but add "..." to indicate it's truncated
        first_words = ' '.join(paragraph_content.split()[:5])
        alt_text = f"{first_words}..."

        # Download image locally
        image_filename = f"image_paragraph_{images_added+1}.png"
        image_local_path = os.path.join("images", image_filename)
        output_image_dir = os.path.join(os.path.dirname(output_file), "images")
        os.makedirs(output_image_dir, exist_ok=True)
        full_image_path = os.path.join(output_image_dir, image_filename)
        if download_image(image_url, full_image_path):
            image_markdown = f"\n\n![{alt_text}](images/{image_filename})\n\n"
        else:
            image_markdown = f"\n\n![{alt_text}]({image_url})\n\n"

        # Insert image before the paragraph
        updated_content = (
            updated_content[:start_pos] +
            image_markdown +
            updated_content[start_pos:]
        )
        images_added += 1
        # Adjust positions for next insertions
        for i in range(len(sorted_paragraphs)):
            if sorted_paragraphs[i][0] > start_pos:
                sorted_paragraphs[i] = (sorted_paragraphs[i][0] + len(image_markdown), sorted_paragraphs[i][1])

    return updated_content, images_added

def download_image(url, file_path, max_retries=2):
    """
    Download an image from a URL and save it to the specified file path.
    Implements retry logic and better error handling.
    Returns True if successful, False otherwise.
    """
    import requests
    
    # Don't try to re-encode the URL - use it as is
    clean_url = url
    
    # Try downloading with retries
    for attempt in range(max_retries + 1):
        try:
            print(f"Attempting to download image from: {clean_url}")
            response = requests.get(clean_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Verify it's actually an image by checking content type
            if 'image' not in response.headers.get('Content-Type', ''):
                print(f"Warning: Response is not an image (attempt {attempt+1})")
                if attempt < max_retries:
                    print(f"Retrying download...")
                    continue
                return False
            
            # Save the image
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Successfully downloaded image to: {file_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            if "500" in str(e):
                print(f"Server error (500) on attempt {attempt+1}: {e}")
            else:
                print(f"Error downloading image (attempt {attempt+1}): {e}")
            
            if attempt < max_retries:
                print(f"Retrying download...")
                
                # On failure, try a simpler URL with fewer parameters
                if "?" in clean_url:
                    clean_url = clean_url.split("?")[0]
                    
                    # Add minimal parameters for size
                    clean_url += "?width=400&height=300&nologo=true"
            else:
                return False
                
    return False

def process_markdown_file(input_file, output_file, max_images=5, process_tables_flag=False):
    """
    Process a single markdown file, adding images to appropriate sections, paragraphs, and tables.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    updated_content = content
    total_images_added = 0
    
    # Process tables if enabled
    if process_tables_flag:
        updated_content, table_images_added = process_tables(updated_content, output_file, max_images)
        total_images_added += table_images_added
        max_images -= table_images_added
        
        if table_images_added > 0:
            print(f"Added images to {table_images_added} table rows.")
    
    # Check if we've reached the maximum number of images
    if max_images <= 0:
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return total_images_added
    
    # Process standalone paragraphs
    updated_content, paragraph_images_added = process_paragraphs(updated_content, output_file, max(1, max_images // 3))
    total_images_added += paragraph_images_added
    max_images -= paragraph_images_added
    
    if paragraph_images_added > 0:
        print(f"Added images to {paragraph_images_added} significant paragraphs.")
    
    # Check if we've reached the maximum number of images
    if max_images <= 0:
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return total_images_added
    
    # Split into sections
    sections = split_markdown_into_sections(updated_content)
    
    # Find the main title (first highest level heading)
    min_heading_level = min([level for _, _, level in sections]) if sections else 0
    main_title = None
    for heading, _, level in sections:
        if level == min_heading_level:
            main_title = heading.group(2)
            break
    
    # Process each section and add images where appropriate
    section_images_added = 0
    for i, (heading, content, level) in enumerate(sections):
        # Skip if we've reached the maximum number of images
        if section_images_added >= max_images:
            break
        
        # Check if we should add an image to this section
        if should_add_image_to_section(content, level):
            heading_text = heading.group(2)
            
            # Generate an image prompt based on section content and heading
            image_prompt = generate_advanced_image_prompt(content, heading_text)
            encoded_prompt = urllib.parse.quote(image_prompt)
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            
            # Use the heading text as the alt text
            alt_text = re.sub(r'\*\*|\*|__|\^|~|`', '', heading_text)
            
            # Download image locally
            image_filename = f"image_section_{section_images_added+1}.png"
            output_image_dir = os.path.join(os.path.dirname(output_file), "images")
            os.makedirs(output_image_dir, exist_ok=True)
            full_image_path = os.path.join(output_image_dir, image_filename)
            
            if download_image(image_url, full_image_path):
                # Insert the image into the section
                updated_section = insert_image_into_section(heading, content, image_filename, alt_text)
                updated_content = updated_content.replace(heading_text + content, heading_text + updated_section)
                section_images_added += 1
                print(f"Added image for section: {alt_text}")
            else:
                print(f"Failed to download image for section: {alt_text}")
    
    total_images_added += section_images_added
    
    # Write the updated content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    return total_images_added

def process_directory(input_dir, output_dir, max_images=5, process_tables_flag=False):
    """
    Process all markdown files in a directory.
    Each output file gets a unique date prefix (incremented by 1 day per file).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_files_processed = 0
    total_images_added = 0

    # Sort files for consistent date assignment
    md_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.md')])
    base_date = datetime(2025, 5, 1).date()  # Start from 2025-05-01

    for idx, file in enumerate(md_files):
        input_file = os.path.join(input_dir, file)
        # Assign a unique date for each file
        file_date = base_date + timedelta(days=idx)
        date_prefix = file_date.strftime("%Y-%m-%d")
        output_filename = f"{date_prefix}-{file}"
        output_file = os.path.join(output_dir, output_filename)
        
        print(f"Processing file: {file}")
        images_added = process_markdown_file(input_file, output_file, max_images, process_tables_flag)
        
        print(f"Added {images_added} images to {output_filename}")
        total_files_processed += 1
        total_images_added += images_added

    print(f"\nSummary:")
    print(f"Processed {total_files_processed} files")
    print(f"Added {total_images_added} images in total")

def main():
    parser = argparse.ArgumentParser(description="Content-Based Markdown Image Generator")
    parser.add_argument("--input", required=True, help="Input markdown file or directory")
    parser.add_argument("--output", required=True, help="Output markdown file or directory")
    parser.add_argument("--max_images", type=int, default=5, help="Maximum number of images to add per file")
    parser.add_argument("--table_images", action="store_true", help="Add images to markdown tables")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_dir():
        process_directory(args.input, args.output, args.max_images, args.table_images)
    else:
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
        images_added = process_markdown_file(args.input, args.output, args.max_images, args.table_images)
        print(f"Added {images_added} images to the markdown file")

if __name__ == "__main__":
    main()