#!/usr/bin/env python
import os
import sys
# ensure project root on PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import precision_recall_fscore_support

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_CSV    = "data/relationlabels.csv"
MODEL_DIR   = "models/relationtagger"
PRETRAINED  = "bert-base-uncased"
BATCH_SIZE  = 16
LR          = 3e-5
EPOCHS      = 5
WARMUP_FRAC = 0.1
MAX_LEN     = 128
DEV_SPLIT   = 0.1
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2ID = {"attack":0, "support":1, "none":2}

# ── Dataset ────────────────────────────────────────────────────────────────────
class RelationDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.a_texts = df["sentence1"].tolist()
        self.b_texts = df["sentence2"].tolist()
        self.labels  = [LABEL2ID[l.strip().lower()] for l in df["label"]]
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

def compute_macro_f1(model, loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            out   = model(ids, attention_mask=mask).logits
            preds.extend(torch.argmax(out, dim=1).cpu().tolist())
            true.extend(batch["labels"].tolist())
    _, _, f1s, _ = precision_recall_fscore_support(true, preds, labels=[0,1,2], average=None)
    return f1s.mean()

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Loading data from {DATA_CSV} and tokenizer {PRETRAINED}")
    df = pd.read_csv(DATA_CSV)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    full_ds = RelationDataset(df, tokenizer)

    # split train/dev
    n_dev = int(DEV_SPLIT * len(full_ds))
    train_ds, dev_ds = random_split(full_ds, [len(full_ds)-n_dev, n_dev],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED, num_labels=3
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0.0
    print(f"Starting training for {EPOCHS} epochs on {DEVICE}")
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch}/{EPOCHS}")
        for i, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            out   = model(ids, attention_mask=mask, labels=labels)
            loss  = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if i % 20 == 0:
                print(f"  Batch {i}/{len(train_loader)}  loss={running_loss/i:.4f}")

        val_f1 = compute_macro_f1(model, dev_loader, DEVICE)
        print(f"Validation macro-F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"→ Saved best model (F1={best_f1:.4f}) to {MODEL_DIR}")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()