from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load your newly fine-tuned model & tokenizer
repo_dir = "models/stancetagger"
model     = AutoModelForSequenceClassification.from_pretrained(repo_dir, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(repo_dir)

# 2. Define the Hub repo ID you originally used
hf_repo_id = "iqasimz/logarg-stancetagger"

# 3. Push to the Hub (this will overwrite the existing files)
model.push_to_hub(hf_repo_id)
tokenizer.push_to_hub(hf_repo_id)

print(f"Pushed updated model & tokenizer to https://huggingface.co/{hf_repo_id}")