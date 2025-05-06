import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Claim/Premise Tagger Setup ─────────────────────────────────────────────────
@st.cache_resource
def load_cp_model():
    repo_id = "iqasimz/logarg-stancetagger"
    tok = DistilBertTokenizerFast.from_pretrained(repo_id)
    mod = DistilBertForSequenceClassification.from_pretrained(repo_id).eval()
    return tok, mod

claim_tok, claim_mod = load_cp_model()

# ── Streamlit UI ────────────────────────────────────────────────────────────────
st.title("Logarg: Claim vs. Premise Tagger")
st.markdown(
    "Enter one sentence per line, then click **Analyze** to see the model’s predictions."
)

input_text = st.text_area("Sentences:", height=200)
if st.button("Analyze"):
    sents = [s.strip() for s in input_text.splitlines() if s.strip()]
    if not sents:
        st.error("Please enter at least one sentence.")
    else:
        # Tokenize & predict
        enc = claim_tok(sents, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = claim_mod(**enc).logits
            probs  = torch.softmax(logits, dim=1)

        # Build results table
        records = []
        for sent, prob in zip(sents, probs):
            p_c, p_p = prob[0].item(), prob[1].item()
            label    = "Claim" if p_c > p_p else "Premise"
            records.append({
                "Sentence":   sent,
                "P(Claim)":   f"{p_c:.3f}",
                "P(Premise)": f"{p_p:.3f}",
                "Label":      label
            })

        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True)