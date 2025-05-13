#!/usr/bin/env python3
import pandas as pd
import sys

def find_nan_rows(csv_path):
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Boolean mask of rows with any NaN
    nan_mask = df.isnull().any(axis=1)
    
    # Extract those rows
    nan_rows = df[nan_mask]
    
    if nan_rows.empty:
        print(f"No rows with NaN values found in {csv_path}.")
    else:
        print(f"Found {len(nan_rows)} rows with NaN values in {csv_path}:\n")
        # Print their indices and which columns are NaN
        for idx, row in nan_rows.iterrows():
            nan_cols = row.index[row.isnull()].tolist()
            print(f"  • Row {idx+1} (zero-based idx {idx}): missing in columns {nan_cols}")
            print(f"    → {row.to_dict()}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_nans.py path/to/relationlabels.csv")
        sys.exit(1)
    find_nan_rows(sys.argv[1])