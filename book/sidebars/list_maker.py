import os
import re
from datetime import datetime

PREFIX = """
"""

ITEM_NAME = "Sidebars"
OUTPUT = 'README.md'
EXCLUDE_DIRS = ['images']
ROOT_DIR = '.'
TYPOS = ['wip', 'obejct', 'orientatation']
DELIMITERS = ['-', '_']


def format_title(filename):
    # Remove the file extension and replace dashes with spaces
    name_without_ext = os.path.splitext(filename)[0].replace('-', ' ')
    # Capitalize the first letter of each word
    return name_without_ext.title()


def generate_markdown_list(dir_path):
    sections = {}
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        section = os.path.relpath(root, dir_path)
        delimiters_pattern = '|'.join(map(re.escape, DELIMITERS))
        words_in_dir = re.split(delimiters_pattern, section.lower())
        if any(word in TYPOS for word in words_in_dir):
            continue
        if section == '.' or section.lower() in TYPOS:
            continue
        section = section.replace('_', ' ').title()
        for delimiter in DELIMITERS:
            section = section.replace(delimiter, ' ')
        items = []
        for file in files:
            words_in_file = re.split(delimiters_pattern, file.lower())
            if any(word in TYPOS for word in words_in_file):
                continue
            if not file.startswith('.') and file.endswith('.md'):
                formatted_title = format_title(file)
                for delimiter in DELIMITERS:
                    formatted_title = formatted_title.replace(delimiter, ' ')
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, dir_path).replace('\\', '/')
                item_str = f"- [{formatted_title}]({relative_path})"
                items.append(item_str)
        if items:
            items.sort()
            sections[section] = items

    markdown_list = []
    for section in sorted(sections.keys()):
        section_str = f"## {section}\n"
        markdown_list.append(section_str)
        print('\n', section_str)
        print('\n'.join(sections[section]))
        markdown_list.extend(sections[section])  # Add sorted items
        markdown_list.append('\n')  # Add newline after each section for spacing

    return '\n'.join(markdown_list)  # Join all markdown entries

print(f"Making a list of {ITEM_NAME} in {OUTPUT}...")

with open(OUTPUT, 'w') as f:
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    f.write(PREFIX)
    f.write(f"\nAuto-generated list of {ITEM_NAME}: {date_time}\n\n")
    f.write(f"# {ITEM_NAME}\n\n")  # Add two newlines for spacing after the title
    markdowns = generate_markdown_list(ROOT_DIR)
    f.write(markdowns)  # Write the markdown content