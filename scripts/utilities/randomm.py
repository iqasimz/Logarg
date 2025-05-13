import csv
import random

# Function to shuffle CSV rows
def shuffle_csv(input_filename, output_filename):
    with open(input_filename, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip the header if present
        rows = list(reader)

    # Shuffle the rows randomly
    random.shuffle(rows)

    # Write the shuffled rows to a new CSV file
    with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header first
        writer.writerows(rows)  # Write the shuffled rows

# Replace 'input.csv' with your input file path and 'shuffled_output.csv' with your desired output file name
input_filename = 'data/relationlabels.csv'  # Input CSV file path
output_filename = 'data/relationlabels.csv'  # Output CSV file path

shuffle_csv(input_filename, output_filename)