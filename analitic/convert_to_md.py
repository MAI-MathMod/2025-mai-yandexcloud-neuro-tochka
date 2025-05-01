import json
import os
from typing import List, Dict, Any

def convert_to_markdown(messages: List[Dict[str, Any]]) -> str:
    markdown_content = []
    
    for message in messages:
        if message.get('type') == 'message':
            # Get message ID
            message_id = message.get('from_id', 'N/A')
            
            # Get date
            date = message.get('date', 'N/A')
            
            # Combine all text entities into a single text
            text = ' '.join(entity['text'] for entity in message.get('text_entities', []) if entity['text'].strip())
            
            # Create markdown entry
            markdown_entry = f"### ID: {message_id}\n"
            markdown_entry += f"**Date:** {date}\n"
            markdown_entry += f"**Text:** {text}\n\n"
            
            markdown_content.append(markdown_entry)
    
    return '\n'.join(markdown_content)

def process_file(input_file: str, output_dir: str):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    # Convert to markdown
    markdown_content = convert_to_markdown(messages)
    
    # Create output filename
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f'{base_filename}.md')
    
    # Write markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

def main():
    input_dir = 'processed_data'
    output_dir = 'markdown_data'
    
    # Process each file
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            process_file(input_file, output_dir)

if __name__ == '__main__':
    main() 