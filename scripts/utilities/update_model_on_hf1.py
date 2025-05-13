#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local checkpoint and HF repo ID
LOCAL_DIR = "models/contagger"
REPO_ID   = "iqasimz/contagger"

# Load your fine-tuned model & tokenizer
model     = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)

# Push to the existing Hugging Face repo
model.push_to_hub(REPO_ID, create_repo=True)
tokenizer.push_to_hub(REPO_ID, create_repo=True)

print(f"✅ Pushed updated model & tokenizer to https://huggingface.co/{REPO_ID}")