#!/usr/bin/env python
import os
import sys
# ensure project root on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ── Config ──────────────────────────────────────────────────────────────────────
TEST_CSV   = "data/relationlabels.csv"      # or point to your held-out test file
MODEL_DIR  = "models/relationtagger"
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# case-insensitive label mapping
LABEL2ID = {"attack": 0, "support": 1, "none": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_test_dataset(csv_path, tokenizer, max_len=128):
    df = pd.read_csv(csv_path)
    a_texts = df["sentence1"].tolist()
    b_texts = df["sentence2"].tolist()
    labels  = [LABEL2ID[l.strip().lower()] for l in df["label"]]

    enc = tokenizer(
        a_texts, b_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(labels, dtype=torch.long)
    )

def main():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    # Prepare data loader
    test_ds     = load_test_dataset(TEST_CSV, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in test_loader:
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            outputs   = model(input_ids=input_ids, attention_mask=attn_mask).logits
            preds     = torch.argmax(outputs, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Save misclassified pairs
    orig = pd.read_csv(TEST_CSV)
    orig["true_norm"] = orig["label"].str.strip().str.lower()
    orig["pred_label"] = [ID2LABEL[p] for p in all_preds]
    orig["pred_norm"]  = orig["pred_label"].str.lower()

    mis = orig[orig["true_norm"] != orig["pred_norm"]]
    out = mis[["sentence1", "sentence2", "label", "pred_label"]].rename(
        columns={"label": "true_label"}
    )
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/relation_wrong.csv", index=False)
    print(f"Misclassified examples saved to data/relation_wrong.csv ({len(out)} rows)")

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0,1,2], average=None
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    print(f"\nTest Accuracy: {acc:.4f}\n")
    for idx in [0,1,2]:
        lbl = ID2LABEL[idx]
        print(f"{lbl:7s}  Precision: {prec[idx]:.4f}  Recall: {rec[idx]:.4f}  F1: {f1[idx]:.4f}")
    print(f"\nMacro F1: {f1_macro:.4f}")

if __name__ == "__main__":
    main()