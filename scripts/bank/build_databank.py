#!/usr/bin/env python
import os
import json
import pandas as pd

# ── Configuration ───────────────────────────────────────────────────────────────
INPUT_CSV    = "data/rawbank.csv"    # raw file: each row [argument; stance]
OUTPUT_JSONL = "data/databank.jsonl" # structured output in JSONL
TOPIC        = "nuclear energy"      # fixed topic tag
# ── End Configuration ───────────────────────────────────────────────────────────

def main():
    # 1. Read the raw CSV (no header, two columns, semicolon-separated)
    #    Column 0: argument text
    #    Column 1: stance ("pro" or "con")
    df = pd.read_csv(
        INPUT_CSV,
        sep=';',                        # use semicolon as separator
        header=None,
        names=["argument", "stance"]
    )

    # 2. Strip any enclosing quotes (straight or curly) from arguments
    df["argument"] = df["argument"].str.strip().str.strip('"“”')

    # 3. Assign sequential argument numbers
    df.insert(0, "argument_number", range(1, len(df) + 1))

    # 4. Add the topic column
    df.insert(1, "topic", TOPIC)

    # 5. Normalize stance to lowercase
    df["stance"] = df["stance"].str.strip().str.lower()

    # 6. Read existing JSONL if exists
    existing = []
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, encoding="utf-8") as f_old:
            existing = [json.loads(line) for line in f_old]

    # 7. Collect new records
    new_records = df.to_dict(orient="records")

    # 8. Prepend new_records to existing
    combined = new_records + existing

    # 9. Write combined records back to JSONL without escaping Unicode
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in combined:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    print(f"Databank created: {OUTPUT_JSONL} ({len(df)} entries)")

if __name__ == "__main__":
    main()
