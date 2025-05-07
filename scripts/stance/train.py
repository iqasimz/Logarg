#!/usr/bin/env python
import os
import sys

# ensure project root on PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_CSV    = "data/labels.csv"
MODEL_DIR   = "models/stancetagger"
PRETRAINED  = MODEL_DIR  # start from your existing checkpoint
BATCH_SIZE  = 16
LR          = 2e-5
EPOCHS      = 5
DEV_SPLIT   = 0.1
WARMUP_FRAC = 0.1
MAX_LEN     = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# case‐insensitive mapping
LABEL2ID = {"claim": 0, "premise": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── Dataset ────────────────────────────────────────────────────────────────────
class StanceDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts  = df["sentence"].tolist()
        self.labels = [LABEL2ID[l.strip().lower()] for l in df["label"]]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc  = self.tokenizer(
            text,
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

def compute_macro_f1(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            out   = model(ids, attention_mask=mask).logits
            preds.extend(torch.argmax(out, dim=1).cpu().tolist())
            trues.extend(batch["labels"].tolist())
    _, _, f1s, _ = precision_recall_fscore_support(trues, preds, labels=[0,1], average=None)
    return f1s.mean()

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Loading data from {DATA_CSV} and checkpoint {PRETRAINED}")
    df = pd.read_csv(DATA_CSV)

    # Stratified split
    train_df, dev_df = train_test_split(
        df,
        test_size=DEV_SPLIT,
        stratify=df["label"].str.strip().str.lower(),
        random_state=42
    )
    print(f"  → Train: {len(train_df)} samples; Dev: {len(dev_df)} samples")

    # Prepare tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    train_ds  = StanceDataset(train_df, tokenizer)
    dev_ds    = StanceDataset(dev_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)

    # Load model and freeze base if desired (comment out to fine-tune all)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED, num_labels=2
    )
    # Freeze encoder layers, train only classifier head:
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(DEVICE)

    # Class weights for loss
    label_counts = df["label"].str.strip().str.lower().value_counts()
    weights = torch.tensor(
        [1.0 / label_counts[l] for l in ["claim", "premise"]],
        device=DEVICE
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # Scheduler
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0.0
    print(f"Starting fine‐tuning for {EPOCHS} epochs on {DEVICE}")

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch}/{EPOCHS}")
        for i, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(ids, attention_mask=mask)
            logits  = outputs.logits
            loss    = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if i % 20 == 0:
                print(f"  Batch {i}/{len(train_loader)}  loss={running_loss/i:.4f}")

        # Dev evaluation
        val_f1 = compute_macro_f1(model, dev_loader, DEVICE)
        print(f"Validation macro‐F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"→ Saved best model (F1={best_f1:.4f}) to {MODEL_DIR}")

    print("\nFine‐tuning complete.")

if __name__ == "__main__":
    main()