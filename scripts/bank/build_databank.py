#!/usr/bin/env python
import os
import json
import pandas as pd

# ── Configuration ───────────────────────────────────────────────────────────────
INPUT_CSV    = "data/rawbank.csv"    # raw file: each row [argument, stance]
OUTPUT_JSONL = "data/databank.jsonl"       # structured output in JSONL
TOPIC        = "nuclear energy"            # fixed topic tag
# ── End Configuration ───────────────────────────────────────────────────────────

def main():
    # 1. Read the raw CSV (no header, two columns)
    #    Column 0: argument text
    #    Column 1: stance ("pro" or "con")
    df = pd.read_csv(INPUT_CSV, header=None, names=["argument", "stance"])
    
    # 2. Assign sequential argument numbers
    df.insert(0, "argument_number", range(1, len(df) + 1))
    
    # 3. Add the topic column
    df.insert(1, "topic", TOPIC)
    
    # 4. Normalize stance to lowercase
    df["stance"] = df["stance"].str.strip().str.lower()
    
    # 5. Read existing JSONL if exists
    existing = []
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL) as f_old:
            existing = [json.loads(line) for line in f_old]
    
    # 6. Collect new records
    new_records = df.to_dict(orient="records")
    
    # 7. Prepend new_records to existing
    combined = new_records + existing
    
    # 8. Write combined records back to JSONL
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w") as f:
        for record in combined:
            json.dump(record, f)
            f.write("\n")
    
    print(f"Databank created: {OUTPUT_JSONL} ({len(df)} entries)")

if __name__ == "__main__":
    main()