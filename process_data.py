import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    
    # Replace question marks with prefix
    if text.endswith('?'):
        text = 'у меня есть вопрос ' + text[:-1]
    
    # Remove punctuation except spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def process_message(message: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only required fields
    processed = {
        'id': message.get('id'),
        'type': message.get('type'),
        'date': message.get('date'),
        'from_id': message.get('from_id'),
        'text_entities': []
    }
    
    # Process text entities
    if 'text_entities' in message:
        for entity in message['text_entities']:
            if 'text' in entity:
                # Convert all text to plain text
                entity['text'] = clean_text(entity['text'])
                # Remove formatting information
                entity['type'] = 'plain'
                processed['text_entities'].append(entity)
    
    return processed

def split_data(data: List[Dict[str, Any]], num_parts: int) -> List[List[Dict[str, Any]]]:
    total_messages = len(data)
    part_size = total_messages // num_parts
    remainder = total_messages % num_parts
    
    parts = []
    start_idx = 0
    
    for i in range(num_parts):
        end_idx = start_idx + part_size + (1 if i < remainder else 0)
        parts.append(data[start_idx:end_idx])
        start_idx = end_idx
    
    return parts

def process_file(input_file: str, output_dir: str):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get messages from the data
    messages = data.get('messages', [])
    
    # Process messages
    processed_data = [process_message(msg) for msg in messages]
    
    # Split into 10 parts
    parts = split_data(processed_data, 10)
    
    # Save parts
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    for i, part in enumerate(parts, 1):
        output_file = os.path.join(output_dir, f'{base_filename}_part{i}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(part, f, ensure_ascii=False, indent=2)

def main():
    input_dir = 'original_data'
    output_dir = 'processed_data'
    
    # Process each file
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            process_file(input_file, output_dir)

if __name__ == '__main__':
    main() 