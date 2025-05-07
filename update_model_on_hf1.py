#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load your fine-tuned relation tagger (3 labels)
repo_dir = "models/relationtagger"
model     = AutoModelForSequenceClassification.from_pretrained(repo_dir)
tokenizer = AutoTokenizer.from_pretrained(repo_dir)

# 2. Specify your HF repo ID
hf_repo_id = "iqasimz/logarg-relationtagger"

# 3. Push to the Hub
model.push_to_hub(hf_repo_id)
tokenizer.push_to_hub(hf_repo_id)

print(f"Pushed updated model & tokenizer to https://huggingface.co/{hf_repo_id}")