# Python script to read from a markdown file and create a JSON Lines (JSONL) file for fine-tuning

import json

def create_jsonl_from_md(md_file_path, jsonl_file_path):
    try:
        with open(md_file_path, 'r') as md_file:
            # Read lines from markdown file
            lines = md_file.readlines()

            with open(jsonl_file_path, 'w') as jsonl_file:
                # Process each line and write as a JSON object to the JSONL file
                for line in lines:
                    # Strip newline characters and any markdown formatting
                    text = line.strip().lstrip('- ')
                    if text:
                        # Create a dictionary and write it as a JSON object
                        json.dump({"text": text}, jsonl_file)
                        jsonl_file.write('\n')
        return "JSONL file created successfully."
    except Exception as e:
        return f"An error occurred: {e}"

# File paths
md_file_path = './custom-dataset.md'
jsonl_file_path = './custom-dataset.jsonl'

create_jsonl_from_md(md_file_path, jsonl_file_path)

