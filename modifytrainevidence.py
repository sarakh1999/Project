def modify_file_quotes_to_braces(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
            open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            # Trim whitespace, then replace the first and last character
            # if they are quotes
            trimmed_line = line.strip()
            if trimmed_line.startswith('"') and trimmed_line.endswith('"'):
                modified_line = '{' + trimmed_line[1:-1] + '}\n'
                output_file.write(modified_line)
            else:
                # Write the line unmodified if it doesn't match the criteria
                output_file.write(line)

# Specify the path to your original file and a temporary new file
input_file_path = 'train_evidence.jsonl'  # Replace with your file path
output_file_path = 'train_evidence_modified.jsonl'  # This will be the modified file

# Modify the file
modify_file_quotes_to_braces(input_file_path, output_file_path)

# If you want to replace the original file with the new one, you can use:
import os
os.replace(output_file_path, input_file_path)
