#!/usr/bin/env python
import os
import sys
import argparse
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

# ── Dataset ─────────────────────────────────────────────────────────────────────
class RelationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts_a = df["sentence1"].tolist()
        self.texts_b = df["sentence2"].tolist()
        # normalize labels
        self.labels  = [l.strip().lower() for l in df["label"]]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a, b = self.texts_a[idx], self.texts_b[idx]
        enc = self.tokenizer(
            a, b,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        label_id = {"attack":0, "support":1, "none":2}[self.labels[idx]]
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label_id, dtype=torch.long)
        }

# ── Compute Macro-F1 for Validation ─────────────────────────────────────────────
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
    _, _, f1s, _ = precision_recall_fscore_support(trues, preds, labels=[0,1,2], average=None)
    return f1s.mean()

# ── Main Training Script ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("Train relation tagger")
    parser.add_argument("--csv",    default="data/relationlabels.csv",
                        help="Path to relationlabels.csv")
    parser.add_argument("--batch",  type=int,   default=16)
    parser.add_argument("--epochs", type=int,   default=5)
    parser.add_argument("--lr",     type=float, default=3e-5)
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Fraction of total steps to warm up")
    parser.add_argument("--maxlen", type=int,   default=128)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--output", default="models/relation-tagger",
                        help="Where to save the trained model")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # Load & split dataset
    df = pd.read_csv(args.csv)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    full_ds = RelationDataset(df, tokenizer, max_len=args.maxlen)
    n_dev   = int(0.1 * len(full_ds))
    train_ds, dev_ds = random_split(full_ds, [len(full_ds)-n_dev, n_dev],
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch)

    # Initialize model & optimizer & scheduler
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0.0
    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        model.train()
        running_loss = 0.0

        for idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labels= batch["labels"].to(device)

            out   = model(ids, attention_mask=mask, labels=labels)
            loss  = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if idx % 20 == 0:
                print(f"  Batch {idx}/{len(train_loader)}  loss={running_loss/idx:.4f}")

        # Validate
        val_f1 = compute_macro_f1(model, dev_loader, device)
        print(f"Validation Macro-F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"→ Saved best model (F1={best_f1:.4f}) to {args.output}")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()