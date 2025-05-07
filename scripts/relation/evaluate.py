#!/usr/bin/env python
import os
import sys
# ensure project root on PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ── Constants ──────────────────────────────────────────────────────────────────
TEST_CSV   = "data/relationlabels.csv"      # or your hold‐out test file
MODEL_DIR  = "models/relationtagger"
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2ID = {"attack":0, "support":1, "none":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

# ── Data Loader ─────────────────────────────────────────────────────────────────
def load_dataset(csv_path, tokenizer):
    df = pd.read_csv(csv_path)
    a_texts = df["sentence1"].tolist()
    b_texts = df["sentence2"].tolist()
    labels  = [LABEL2ID[l.strip().lower()] for l in df["label"]]

    enc = tokenizer(
        a_texts, b_texts,
        padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    )
    return TensorDataset(
        enc["input_ids"], enc["attention_mask"], torch.tensor(labels)
    )

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    print(f"Loading test data from {TEST_CSV}")
    ds = load_dataset(TEST_CSV, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            logits    = model(input_ids=input_ids, attention_mask=attn_mask).logits
            preds     = torch.argmax(logits, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Misclassified
    orig = pd.read_csv(TEST_CSV)
    orig["true_norm"]  = orig["label"].str.strip().str.lower()
    orig["pred_label"] = [ID2LABEL[p] for p in all_preds]
    orig["pred_norm"]  = orig["pred_label"].str.lower()

    mis = orig[orig["true_norm"] != orig["pred_norm"]]
    out = mis[["sentence1","sentence2","label","pred_label"]].rename(
        columns={"label":"true_label"}
    )
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/relation_wrong.csv", index=False)
    print(f"Saved misclassified pairs to data/relation_wrong.csv ({len(out)} rows)")

    # Metrics
    acc   = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0,1,2], average=None
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    print(f"\nTest Accuracy: {acc:.4f}")
    for idx in [0,1,2]:
        lbl = ID2LABEL[idx]
        print(f"{lbl:7s}  P:{prec[idx]:.4f}  R:{rec[idx]:.4f}  F1:{f1[idx]:.4f}")
    print(f"\nMacro F1: {f1_macro:.4f}")

if __name__ == "__main__":
    main()