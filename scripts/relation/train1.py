#!/usr/bin/env python
import os
import sys
import time
import numpy as np
from tqdm import tqdm

# ensure project root on PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

# ── Constants ───────────────────────────────────────────────────────────────────
DATA_CSV       = "data/relationlabels.csv"
MODEL_DIR      = "models/relationtagger"
PRETRAINED     = "bert-base-uncased"
BATCH_SIZE     = 16
LR             = 3e-5
EPOCHS         = 8
WARMUP_FRAC    = 0.1
MAX_LEN        = 128
NUM_FOLDS      = 5
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2ID = {"attack": 0, "support": 1, "none": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── Dataset ─────────────────────────────────────────────────────────────────────
class RelationDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.a_texts  = df["sentence1"].tolist()
        self.b_texts  = df["sentence2"].tolist()
        self.labels   = [LABEL2ID[l.strip().lower()] for l in df["label"]]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a, b = self.a_texts[idx], self.b_texts[idx]
        enc = self.tokenizer(
            a, b,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Evaluation ──────────────────────────────────────────────────────────────────
def compute_classwise_f1(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            out   = model(ids, attention_mask=mask).logits
            preds.extend(torch.argmax(out, dim=1).cpu().tolist())
            trues.extend(batch["labels"].tolist())
    _, _, f1s, _ = precision_recall_fscore_support(trues, preds, labels=[0,1,2], average=None)
    return f1s

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    df.dropna(subset=["sentence1", "sentence2", "label"], inplace=True)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    dataset = RelationDataset(df, tokenizer)
    targets = [LABEL2ID[l.strip().lower()] for l in df["label"]]

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_f1s = []
    best_fold = -1
    best_macro_f1 = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

        model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED, num_labels=3)
        model.to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=LR)
        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_f1s = None
        best_f1 = 0.0

        for epoch in range(1, EPOCHS+1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                ids    = batch["input_ids"].to(DEVICE)
                mask   = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

            f1s = compute_classwise_f1(model, val_loader, DEVICE)
            macro_f1 = f1s.mean()
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_f1s = f1s
                fold_model_dir = os.path.join(MODEL_DIR, f"fold{fold+1}")
                os.makedirs(fold_model_dir, exist_ok=True)
                model.save_pretrained(fold_model_dir)
                tokenizer.save_pretrained(fold_model_dir)

        print(f"\nFold {fold+1} best checkpoint F1 scores:")
        for i, score in enumerate(best_f1s):
            print(f"  {ID2LABEL[i]}: {score:.4f}")

        fold_f1s.append(best_f1s)
        if best_f1 > best_macro_f1:
            best_macro_f1 = best_f1
            best_fold = fold + 1

    print(f"\nBest performing fold: Fold {best_fold}")

if __name__ == "__main__":
    main()
