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
    # 1. Read new entries from the raw CSV (semicolon-separated)
    df_new = pd.read_csv(
        INPUT_CSV,
        sep=';',
        header=None,
        names=["argument", "stance"]
    )

    # 2. Strip enclosing straight or curly quotes
    df_new["argument"] = df_new["argument"].str.strip().str.strip('"“”')

    # 3. Normalize stance and add topic
    df_new["stance"] = df_new["stance"].str.strip().str.lower()
    df_new.insert(1, "topic", TOPIC)

    # 4. Load existing JSONL if present
    if os.path.exists(OUTPUT_JSONL):
        existing = pd.DataFrame([
            json.loads(line) for line in open(OUTPUT_JSONL, encoding="utf-8")
        ])
        # Determine starting argument number
        max_num = existing["argument_number"].max()
    else:
        existing = pd.DataFrame(columns=["argument_number", "topic", "argument", "stance"])
        max_num = 0

    # 5. Assign argument numbers to new entries sequentially after existing ones
    df_new.insert(0, "argument_number", range(max_num + 1, max_num + 1 + len(df_new)))

    # 6. Combine existing and new entries (existing first, then new)
    combined = pd.concat([existing, df_new], ignore_index=True)

    # 7. Write combined records back to JSONL without escaping Unicode
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in combined.to_dict(orient="records"):
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    print(f"Databank updated: {OUTPUT_JSONL} ({len(combined)} total entries)")

if __name__ == "__main__":
    main()
