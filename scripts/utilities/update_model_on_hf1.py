#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local checkpoint and HF repo ID
LOCAL_DIR = "models/relationtagger"
REPO_ID   = "iqasimz/logarg-relationtagger"

# Load your fine-tuned model & tokenizer
model     = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)

# Push to the existing Hugging Face repo
model.push_to_hub(REPO_ID)
tokenizer.push_to_hub(REPO_ID)

print(f"✅ Pushed updated model & tokenizer to https://huggingface.co/{REPO_ID}")