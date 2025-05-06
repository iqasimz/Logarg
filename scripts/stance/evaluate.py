#!/usr/bin/env python
import os
import sys
# Add project root to imports
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Config
TEST_CSV   = "data/test.csv"
MODEL_DIR  = "models/stancetagger"
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL2ID   = {"Claim": 0, "Premise": 1}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}

def load_test_dataset(csv_path, tokenizer, max_len=128):
    df = pd.read_csv(csv_path)
    texts  = df["sentence"].tolist()
    labels = [LABEL2ID[l] for l in df["label"]]

    enc = tokenizer(
        texts,
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
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
    model.to(DEVICE)
    model.eval()

    # Prepare data
    test_ds     = load_test_dataset(TEST_CSV, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in test_loader:
            input_ids  = input_ids.to(DEVICE)
            attn_mask  = attn_mask.to(DEVICE)
            outputs    = model(input_ids=input_ids, attention_mask=attn_mask)
            logits     = outputs.logits
            preds      = torch.argmax(logits, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Save misclassified examples
    orig_df = pd.read_csv(TEST_CSV)
    orig_df['pred_label'] = [ID2LABEL[p] for p in all_preds]
    mis_df = orig_df[orig_df['pred_label'] != orig_df['label']]
    df_out = mis_df[['sentence', 'label', 'pred_label']].rename(columns={'label': 'true_label'})
    os.makedirs(os.path.dirname('data/wrong.csv'), exist_ok=True)
    df_out.to_csv('data/wrong.csv', index=False)
    print(f"Misclassified examples saved to data/wrong.csv ({len(df_out)} rows)")

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    prec, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0,1], average=None
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    # Display
    print(f"Test Accuracy:  {accuracy:.4f}\n")
    for idx, label in ID2LABEL.items():
        print(f"{label:7s}  Precision: {prec[idx]:.4f}  Recall: {recall[idx]:.4f}  F1: {f1[idx]:.4f}")
    print(f"\nMacro F1:       {f1_macro:.4f}")

if __name__ == "__main__":
    main()