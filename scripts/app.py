import streamlit as st
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ── Claim/Premise Tagger Setup ─────────────────────────────────────────────────
@st.cache_resource
def load_cp_model():
    repo_id = "iqasimz/logarg-stancetagger"
    tok = DistilBertTokenizerFast.from_pretrained(repo_id)
    mod = DistilBertForSequenceClassification.from_pretrained(repo_id).eval()
    return tok, mod

# ── Relation Tagger Setup ───────────────────────────────────────────────────────
@st.cache_resource
def load_rel_model():
    repo_id = "iqasimz/logarg-relationtagger"
    tok = AutoTokenizer.from_pretrained(repo_id)
    mod = AutoModelForSequenceClassification.from_pretrained(repo_id).eval()
    return tok, mod

claim_tok, claim_mod = load_cp_model()
rel_tok,   rel_mod   = load_rel_model()

REL_LABELS = ["attack", "support", "none"]

# ── Streamlit UI ────────────────────────────────────────────────────────────────
st.title("Logarg")
tabs = st.tabs(["Stance Classifier", "Relation Classifier"])

########## Tab 1: Claim vs Premise ##########
with tabs[0]:
    input_text = st.text_area("Enter one sentence per line:", height=200, key="cp_input")
    if st.button("Analyze", key="cp_button"):
        sents = [s.strip() for s in input_text.splitlines() if s.strip()]
        if not sents:
            st.error("Please enter at least one sentence.")
        else:
            enc = claim_tok(sents, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = claim_mod(**enc).logits
                probs  = torch.softmax(logits, dim=1)

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

########## Tab 2: Relation ##########
with tabs[1]:
    st.markdown("Enter two sentences to see whether the second **supports**, **attacks**, or is **none** in relation to the first.")
    sent1 = st.text_area("Sentence 1:", height=100, key="rel_s1")
    sent2 = st.text_area("Sentence 2:", height=100, key="rel_s2")
    if st.button("Analyze", key="rel_button"):
        if not sent1.strip() or not sent2.strip():
            st.error("Please enter both sentences.")
        else:
            enc = rel_tok([sent1], [sent2], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = rel_mod(**enc).logits
                probs  = torch.softmax(logits, dim=1)[0].tolist()

            # Build display
            df = pd.DataFrame([{
                "Relation":    REL_LABELS[i],
                "Probability": f"{p:.3f}"
            } for i, p in enumerate(probs)])
            st.table(df)

            # Highlight predicted
            best = REL_LABELS[probs.index(max(probs))]
            st.success(f"Predicted relation: **{best.upper()}**")