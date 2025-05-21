import subprocess
import os
import re
import sys
import signal
import string
import docx  # python-docx library for reading .docx files
import time
import json
import pickle
from datetime import datetime

# Global flag for handling interrupts gracefully
terminating = False

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    global terminating
    print("\n[!] Interrupt received. Finishing current chunk and saving progress...")
    terminating = True
    # Don't exit immediately - let the current operation complete and save

def check_ollama_model(model_name="llama3"):
    """
    Check if the Ollama service is running and if the specified model is available.
    """
    try:
        # Check if Ollama service is running
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("[❌] Ollama service is not running. Please start the Ollama service and try again.")
            sys.exit(1)

        # Check if the specified model is available
        if model_name in result.stdout:
            print(f"[✓] '{model_name}' is available in Ollama.")
            return True
        else:
            print(f"[!] Model '{model_name}' is NOT found in Ollama.")
            print("    Please pull the model using: ollama pull llama3")
            return False
    except FileNotFoundError:
        print("[❌] Ollama is not installed or not found in PATH.")
        print("   Please install Ollama from https://ollama.com and try again.")
        sys.exit(1)

def extract_text_from_docx(docx_file):
    """Extract text content from a DOCX file, including tables."""
    try:
        doc = docx.Document(docx_file)
        full_text = []

        # Extract text from paragraphs
        for para in doc.paragraphs:
            if para.text.strip():  # Skip empty paragraphs
                full_text.append(para.text)

        # Extract text from tables - improved table handling
        for table in doc.tables:
            # First check if table has content
            has_content = False
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        has_content = True
                        break
                if has_content:
                    break
            
            if not has_content:
                continue
                
            # Process table as markdown
            table_text = []
            table_text.append("Table content:")
            
            # Get header row
            header_cells = [cell.text.strip() for cell in table.rows[0].cells]
            table_text.append("| " + " | ".join(header_cells) + " |")
            
            # Add separator row
            table_text.append("| " + " | ".join(["---" for _ in header_cells]) + " |")
            
            # Add data rows
            for row in table.rows[1:]:
                row_cells = [cell.text.strip() for cell in row.cells]
                table_text.append("| " + " | ".join(row_cells) + " |")
            
            full_text.append("\n".join(table_text))

        return '\n\n'.join(full_text)
    except Exception as e:
        print(f"[❌] Error extracting text from DOCX: {e}")
        sys.exit(1)

def split_text_into_chunks(raw_text, max_chars=1500):
    """
    Split the raw text into smaller chunks based on a maximum character limit.
    Try to split at logical points like paragraph breaks.
    INCREASED max_chars value to 1500 to ensure more substantial content per chunk.
    """
    if len(raw_text) <= max_chars:
        return [raw_text]
        
    chunks = []
    paragraphs = raw_text.split("\n\n")
    current_chunk = []
    current_length = 0
    
    # If we have tables, try to keep each table in its own chunk if possible
    table_separator = "Table content:"
    
    for paragraph in paragraphs:
        # Check if this is a table
        is_table = table_separator in paragraph
        paragraph_length = len(paragraph) + 2  # +2 for the newlines
        
        # For tables, try to keep them together as a single chunk when possible
        if is_table:
            # If we have content already, save current chunk first
            if current_chunk and current_length > 0:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
    
            # For very large tables, split them into multiple chunks
            if paragraph_length > max_chars * 2:
                # Split large table into smaller parts
                table_lines = paragraph.split("\n")
                header_lines = []
                data_lines = []
        
                # Identify header and data lines
                for line in table_lines:
                    if not "|" in line:
                        header_lines.append(line)
                    elif "---" in line or table_lines.index(line) <= 2:
                        header_lines.append(line)
                    else:
                        data_lines.append(line)
        
                # Create header chunk (including "Table content:" and the header rows)
                header_chunk = "\n".join(header_lines)
        
                # Calculate how many rows per chunk
                rows_per_chunk = max(1, int(max_chars / (len(data_lines[0]) if data_lines else max_chars)))
        
                # Create data chunks with header repeated
                for i in range(0, len(data_lines), rows_per_chunk):
                    end_idx = min(i + rows_per_chunk, len(data_lines))
                    table_chunk = header_chunk + "\n" + "\n".join(data_lines[i:end_idx])
                    chunks.append(table_chunk)
            else:
                # Add smaller table as its own chunk
                chunks.append(paragraph)
            continue
                
        # For regular paragraphs, combine them until we reach max_chars
        if current_length + paragraph_length > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    # Make sure no chunk is empty
    chunks = [c for c in chunks if c.strip()]
    
    # Ensure reasonable chunk sizes - combine very small chunks
    i = 0
    while i < len(chunks) - 1:
        if len(chunks[i]) < max_chars * 0.3 and len(chunks[i + 1]) < max_chars * 0.7:
            # Combine this chunk with the next one if they're both relatively small
            combined = chunks[i] + "\n\n" + chunks[i + 1]
            chunks[i] = combined
            chunks.pop(i + 1)
        else:
            i += 1
    
    # Print debug info on chunks
    print(f"[✓] Split content into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"[*] Chunk {i+1} length: {len(chunk)} characters")
        
    return chunks

def save_checkpoint(chunks_data, output_dir, current_chunk):
    """Save progress checkpoint for resuming later"""
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    
    checkpoint_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": len(chunks_data),
        "processed_chunks": current_chunk + 1,
        "processed_data": chunks_data[:current_chunk+1],
        "remaining_start": current_chunk + 1
    }
    
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"[✓] Checkpoint saved at chunk {current_chunk + 1}")

def load_checkpoint(output_dir):
    """Load checkpoint for resuming processing"""
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    
    if not os.path.exists(checkpoint_file):
        return None, []
    
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        
        print(f"[✓] Found checkpoint from {checkpoint['timestamp']}")
        print(f"[*] {checkpoint['processed_chunks']}/{checkpoint['total_chunks']} chunks were processed")
        
        # Return the start position and already processed outputs
        return checkpoint["remaining_start"], checkpoint["processed_data"]
    except Exception as e:
        print(f"[!] Error loading checkpoint: {e}")
        return None, []

def process_chunk(chunk, chunk_index, total_chunks, is_mostly_table=False):
    """Process a single chunk and return the result"""
    # Determine if this chunk is mostly a table
    if not is_mostly_table:
        is_mostly_table = "Table content:" in chunk and chunk.count("|") > 5
    
    # Simplified prompt for table-heavy content
    if is_mostly_table:
        prompt = f"""
You are creating a detailed course module for a Jekyll website. This chunk contains table data.

Your task is to create a comprehensive markdown document with proper front matter.
The front matter MUST BE at the very beginning of your response and formatted exactly like this:

---
title: "Table Data: Create a Clear, Specific Title"
layout: post
---

Then convert the following raw table content to proper markdown tables and add explanatory content:

{chunk}

Guidelines:
1. Make sure the markdown table has proper headers and formatting
2. Provide substantial explanatory text around the table (at least 300 words)
3. Explain what the table shows and its significance
4. Add relevant context, examples, and applications
5. Add a detailed conclusion summarizing key takeaways

Your response MUST begin with the front matter block (---) and include at least 2-3 paragraphs of text before the table and 2-3 paragraphs after it.
"""
    else:
        # Enhanced prompt for regular content with MORE DETAILED instructions
        prompt = f"""
You are an expert course content creator for a Jekyll website. Your task is to create a detailed, well-structured module from the provided content.

Your markdown document MUST start with front matter exactly like this:

---
title: "Create a Clear, Specific Title"
layout: post
---

Then create comprehensive educational content with proper markdown formatting:

# Main Heading

## Subheading 1
Detailed content here...

## Subheading 2
More detailed content here...

CONTENT REQUIREMENTS:
- Create ROBUST, THOROUGH educational content (minimum 800-1000 words)
- Use a proper hierarchy of headings (# for main, ## for sub-sections)
- Include bullet points and numbered lists where appropriate
- Format any tables correctly
- Add examples, explanations, and applications
- Include a summary or conclusion section

Here's the raw content to transform into a comprehensive module:

{chunk}

IMPORTANT: Your response MUST begin with the front matter block, include multiple sections, and provide SUBSTANTIAL educational content.
"""
    
    max_attempts = 4  # Try up to 4 times
    for attempt in range(max_attempts):
        try:
            print(f"[*] Processing chunk {chunk_index + 1} of {total_chunks} (attempt {attempt+1})...")
            
            # Increase timeout for larger chunks or later attempts
            timeout_duration = 300 + (attempt * 120) # 300s + 120s per retry
            
            result = subprocess.run(
                ['ollama', 'run', 'llama3', prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout_duration
            )
            
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip()
                
                # Print the first 100 characters of the output for debugging
                print(f"[*] Output starts with: {output[:100].replace('\n', ' ')}")
                
                # Check content length - reject if too short
                if len(output) < 500:  # Minimum content length threshold
                    print(f"[!] Generated content too short ({len(output)} chars), retrying...")
                    continue
                
                # Look for front matter
                front_matter_match = re.search(r'---\s*\n(.*?)\n---', output, re.DOTALL)
                
                # If front matter not found, try to fix it
                if not front_matter_match:
                    print(f"[!] No front matter found, adding it...")
                    
                    # Look for a heading to use as title
                    heading_match = re.search(r'#\s+(.+?)(?:\n|$)', output)
                    if heading_match:
                        title = heading_match.group(1).strip()
                    else:
                        title = f"Module {chunk_index + 1}"
                        
                    # Add front matter to the beginning
                    output = f"""---
title: "{title}"
layout: post
---

{output}"""
                
                # Verify we have front matter now
                front_matter_match = re.search(r'---\s*\n(.*?)\n---', output, re.DOTALL)
                if front_matter_match:
                    # Make sure the content has proper headers
                    content_after_front_matter = output.split('---', 2)[2].strip()
                    
                    # If no headers, add some structure
                    if not re.search(r'#\s+', content_after_front_matter):
                        # Extract title from front matter
                        title_match = re.search(r'title:\s*["\']?(.+?)["\']?(?:\n|\r)', output)
                        if title_match:
                            title = title_match.group(1).strip()
                        else:
                            title = f"Module {chunk_index + 1}"
                            
                        # Add proper headings
                        content_after_front_matter = f"# {title}\n\n{content_after_front_matter}\n\n## Summary\n\nThis module covers key concepts related to this topic."
                        
                        # Reconstruct the output
                        output = f"---\n{front_matter_match.group(1)}\n---\n\n{content_after_front_matter}"
                    
                    print(f"[✓] Successfully processed chunk {chunk_index + 1}")
                    return output
                else:
                    print(f"[!] Failed to fix front matter, retrying...")
            else:
                print(f"[❌] Failed to get output from LLaMA 3 (attempt {attempt+1})")
                if result.stderr:
                    print(f"[*] Error output: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"[❌] Timeout expired for chunk {chunk_index + 1}, attempt {attempt + 1}. Retrying with longer timeout...")
        except Exception as e:
            print(f"[❌] Error processing chunk {chunk_index + 1}: {str(e)}")
    
    # If all attempts failed, return a fallback with substantial content
    print(f"[!] All attempts failed for chunk {chunk_index + 1}, creating fallback content")
    
    # Create a more robust fallback with actual content
    fallback_title = f"Module {chunk_index + 1}"
    
    # Try to extract a meaningful title from the chunk
    first_sentence = chunk.split('.')[0][:50] if '.' in chunk else chunk[:50]
    if len(first_sentence) > 10:
        fallback_title = first_sentence + "..."
    
    fallback = f"""---
title: "{fallback_title}"
layout: post
---

# {fallback_title}

## Introduction

This module contains important content related to our course. The raw content is presented below for reference, but we'll be expanding on these topics in class.

## Content Overview

The following content provides key information for this module:

```
{chunk[:300]}{'...' if len(chunk) > 300 else ''}
```

## Main Topics

This section covers the main topics introduced in this module. Some of the key points include:

- Understanding the fundamental concepts
- Applying these concepts in practical scenarios
- Analyzing related case studies
- Evaluating outcomes and results

## Further Study

For a deeper understanding of these topics, consider reviewing the related materials and resources provided in the course bibliography.

## Summary

This module introduced several important concepts that build on our previous discussions. In the next module, we'll explore related topics in more detail.

"""
    return fallback

def save_chunk_to_file(chunk_output, chunk_index, output_dir):
    """Save a single chunk output to its own file immediately"""
    # Make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract title for filename
    front_matter_match = re.search(r'---\s*\n(.*?)\n---', chunk_output, re.DOTALL)
    if front_matter_match:
        title_match = re.search(r'title:\s*["\']?(.+?)["\']?(?:\n|\r)', chunk_output)
        if title_match:
            title = title_match.group(1).strip()
            # Create safe filename
            safe_title = ''.join(c for c in title.lower() if c.isalnum() or c in ' -_')
            safe_title = safe_title.replace(' ', '_')
            if not safe_title:
                safe_title = f"module_{chunk_index+1}"
        else:
            safe_title = f"module_{chunk_index+1}"
    else:
        safe_title = f"module_{chunk_index+1}"
        
    # Make sure the title is ASCII compatible and not too long
    safe_title = ''.join(c for c in safe_title if c in string.printable)
    safe_title = safe_title[:50]  # Limit filename length
        
    # Avoid duplicate filenames
    filename = f"{chunk_index+1:02d}_{safe_title}.md"  # Add numeric prefix for order
    filepath = os.path.join(output_dir, filename)
    counter = 1
    while os.path.exists(filepath):
        filename = f"{chunk_index+1:02d}_{safe_title}_{counter}.md"
        filepath = os.path.join(output_dir, filename)
        counter += 1

    # Write content to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(chunk_output)
    
    print(f"[✓] Created file: {filename}")
    return filename

def process_chunks_with_checkpointing(chunks, output_dir, start_chunk=0, existing_outputs=None):
    """Process all chunks with checkpointing and immediate file saving"""
    global terminating
    
    # Initialize outputs list with existing outputs if available
    if existing_outputs is None or not existing_outputs:
        all_outputs = []
    else:
        all_outputs = existing_outputs
        print(f"[*] Starting with {len(all_outputs)} previously processed chunks")
    
    # Create progress report file to track overall progress
    progress_file = os.path.join(output_dir, "processing_progress.txt")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(f"\nProcessing started/resumed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total chunks: {len(chunks)}, Starting from: {start_chunk+1}\n")
    
    # Process all chunks from the starting point
    try:
        for i in range(start_chunk, len(chunks)):
            # Check if termination was requested
            if terminating:
                print(f"[!] Termination requested, saving checkpoint and exiting after chunk {i}...")
                break
                
            # Process this chunk
            chunk_output = process_chunk(chunks[i], i, len(chunks))
            
            # Save this output immediately
            if chunk_output:
                filename = save_chunk_to_file(chunk_output, i, output_dir)
                
                # Append to all_outputs for the checkpoint
                all_outputs.append(chunk_output)
                
                # Save checkpoint after each chunk
                save_checkpoint(all_outputs, output_dir, i)
                
                # Update progress file
                with open(progress_file, "a", encoding="utf-8") as f:
                    f.write(f"Completed chunk {i+1}/{len(chunks)}: {filename}\n")
            else:
                print(f"[!] No output generated for chunk {i+1}")
                
                # Save a note about the failed chunk in progress file
                with open(progress_file, "a", encoding="utf-8") as f:
                    f.write(f"Failed to process chunk {i+1}/{len(chunks)}\n")
    
    except KeyboardInterrupt:
        print("\n[!] Processing interrupted by user")
        terminating = True
    except Exception as e:
        print(f"[❌] Error during processing: {e}")
        # Update progress file about the error
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(f"Error at chunk {i+1}/{len(chunks)}: {str(e)}\n")
    
    # Create a summary file
    if len(all_outputs) > 0:
        with open(os.path.join(output_dir, "processing_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Processing {'completed' if i == len(chunks)-1 and not terminating else 'interrupted'} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write(f"Processed chunks: {len(all_outputs)}\n")
            f.write(f"Completion: {len(all_outputs)}/{len(chunks)} ({len(all_outputs)*100/len(chunks):.1f}%)\n")
    
    return all_outputs

def compile_index_file(output_dir, outputs):
    """Create an index file listing all generated modules"""
    index_content = f"""---
title: "Course Content Index"
layout: default
---

# Course Content Index

This index was generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Available Modules

"""
    # Get all markdown files in the directory
    md_files = [f for f in os.listdir(output_dir) if f.endswith(".md") and not f.startswith("index") 
                and not f == "processing_summary.md" and not f == "raw_output.md"]
    
    # Sort files by the numeric prefix to maintain proper order
    md_files.sort()
    
    # Extract titles and create links
    for filename in md_files:
        try:
            with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
                title_match = re.search(r'title:\s*["\']?(.+?)["\']?(?:\n|\r)', content)
                if title_match:
                    title = title_match.group(1).strip()
                else:
                    title = filename.replace(".md", "").replace("_", " ").title()
                    
                index_content += f"* [{title}]({filename})\n"
        except Exception as e:
            index_content += f"* [{filename}]({filename}) (Error reading title)\n"
    
    # Save the index file
    with open(os.path.join(output_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write(index_content)
    
    print(f"[✓] Created index file: index.md")

def main():
    # Set up signal handlers for graceful termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check if Ollama and llama3 model are available
    model = "llama3"
    if not check_ollama_model(model):
        sys.exit(1)

    # Get input file path from command line or use default
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "course.docx"  # Default file name

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"[❌] Input file '{input_file}' not found.")
        print(f"    Usage: python {sys.argv[0]} [path_to_docx_file] [output_dir] [start_chunk]")
        sys.exit(1)

    # Get output directory option
    output_dir = "generated_markdown"
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract text from DOCX file
    print(f"[*] Extracting text from '{input_file}'...")
    try:
        raw_text = extract_text_from_docx(input_file)
        
        # Debug: Save raw text for review if needed
        with open(os.path.join(output_dir, "raw_extracted_text.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"[✓] Raw text extracted and saved to '{os.path.join(output_dir, 'raw_extracted_text.txt')}'")
        
        # Split text into chunks - INCREASED chunk size to 1500 chars
        chunks = split_text_into_chunks(raw_text, max_chars=1500)
        
        # Check if we should resume from checkpoint
        resume_from, existing_outputs = load_checkpoint(output_dir)
        
        # Get starting chunk if provided via command line (overrides checkpoint)
        start_chunk = resume_from if resume_from is not None else 0
        if len(sys.argv) > 3:
            try:
                start_chunk = int(sys.argv[3]) - 1  # Convert from 1-based to 0-based indexing
                if start_chunk < 0:
                    start_chunk = 0
                print(f"[*] Will resume processing from chunk {start_chunk + 1} (command line override)")
                # Reset existing outputs if starting from a specific chunk
                existing_outputs = []
            except ValueError:
                print("[!] Invalid start chunk number, using checkpoint or starting from beginning")
        
        # Process chunks with immediate saving and checkpointing
        print(f"[*] Starting processing from chunk {start_chunk + 1}...")
        print("[*] Press Ctrl+C to stop processing (current progress will be saved)")
        
        all_outputs = process_chunks_with_checkpointing(chunks, output_dir, start_chunk, existing_outputs)
        
        # Create an index file of all modules
        compile_index_file(output_dir, all_outputs)
        
        if terminating:
            print(f"[!] Processing terminated. {len(all_outputs)}/{len(chunks)} chunks processed.")
            print(f"[*] You can resume later from chunk {len(all_outputs) + 1} using:")
            print(f"    python {sys.argv[0]} {input_file} {output_dir} {len(all_outputs) + 1}")
        else:
            print(f"[✓] Processing completed! {len(all_outputs)}/{len(chunks)} chunks processed.")
        
        print(f"[✓] Check '{output_dir}' directory for the generated markdown files.")
        print(f"[*] See '{os.path.join(output_dir, 'index.md')}' for an index of all modules.")

    except Exception as e:
        print(f"[❌] An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()