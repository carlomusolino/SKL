import os
import re

# Function to replace 'skl' or 'skl' with 'SKL' while maintaining capitalization
def replace_grace_with_skl(content):
    def repl(match):
        word = match.group(0)
        if word.isupper():
            return "SKL"
        elif word[0].isupper():
            return "Skl"
        else:
            return "skl"

    # Regular expression to match 'skl' or 'skl' with case-insensitivity
    pattern = re.compile(r'\b(skl|skl)\b', re.IGNORECASE)
    return pattern.sub(repl, content)

# Function to recursively search through files and replace 'skl'/'skl'
def process_files_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Process only text files (you can modify this as needed)
            if file.endswith(".txt") or file.endswith(".py") or file.endswith(".cpp"):  # Add more extensions if needed
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace skl/skl with SKL
                new_content = replace_grace_with_skl(content)
                
                # Write the updated content back to the file if changes were made
                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated: {file_path}")

# Replace this with the path to your directory
directory_path = './'

# Run the script
process_files_in_directory(directory_path)
