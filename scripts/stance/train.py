#!/usr/bin/env python
import os
import sys
# Add project root to Python path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# 1. Config
DATA_PATH    = "data/labels.csv"
MODEL_DIR    = "models/stancetagger"
OUTPUT_PATH  = os.path.join(MODEL_DIR, "finetuned.pt")
PRETRAINED   = OUTPUT_PATH if os.path.exists(OUTPUT_PATH) else MODEL_DIR
BATCH_SIZE   = 16
LR           = 2e-5
EPOCHS       = 3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL2ID     = {"Claim": 0, "Premise": 1}

# 2. Dataset
class EdgeCaseDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        df = pd.read_csv(csv_path)
        self.texts  = df["sentence"].tolist()
        self.labels = [LABEL2ID[l] for l in df["label"]]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 3. Prepare
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
dataset   = EdgeCaseDataset(DATA_PATH, tokenizer)
loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED,
    num_labels=2
)
model.to(DEVICE)
model.train()

optimizer = AdamW(model.parameters(), lr=LR)

# 4. Training loop
for epoch in range(1, EPOCHS+1):
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[Epoch {epoch}/{EPOCHS}] avg_loss = {avg_loss:.4f}")

    # Quick eval on edge cases
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            ).logits
            preds = logits.argmax(dim=-1).cpu()
            correct += (preds == batch["labels"]).sum().item()
    acc = correct / len(dataset)
    print(f"→ Edge-case accuracy: {acc:.2%}")
    model.train()

# 5. Save
os.makedirs(MODEL_DIR, exist_ok=True)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Model & tokenizer saved to {MODEL_DIR}")