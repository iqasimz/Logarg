import os
import json
# Allow duplicate OpenMP runtimes on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import torch
import numpy as np
import random
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(layout="centered")

# ── Load Databank ───────────────────────────────────────────────────────────────
def load_databank(path="databank.jsonl"):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["argument_number", "topic", "stance", "argument"])
    records = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                st.warning(f"Skipping malformed JSON on line {lineno}")
    return pd.DataFrame(records)

db = load_databank()

# ── Sidebar: Topic, Stance & Mode ──────────────────────────────────────────────
topics = sorted(db['topic'].unique())
selected_topic = st.sidebar.selectbox("Select debate topic:", topics)
stance = st.sidebar.radio("Your stance on the topic:", ["Pro", "Against"])
mode = st.sidebar.selectbox("Mode", ["Proponent", "Opponent", "Debating Coach"])
filtered_db = db[db['topic'] == selected_topic].reset_index(drop=False)

# ── Load Relation Model & Tokenizer ────────────────────────────────────────────
@st.cache_resource
def load_relation_model(repo):
    tok = AutoTokenizer.from_pretrained(repo)
    mod = AutoModelForSequenceClassification.from_pretrained(repo)
    mod.eval()
    return tok, mod

repo_path = "iqasimz/protagger" if stance == "Pro" else "iqaimz/contagger"
rel_tok, rel_mod = load_relation_model(repo_path)
# label order: 0=attack, 1=support, 2=none
REL_LABELS = ["attack", "support", "none"]

# ── Templates ─────────────────────────────────────────────────────────────────
TEMPLATES = {
    "Opponent": [
        "That’s where I’d have to disagree. {argument}",
        "Here’s a major problem with your view: {argument}",
        "But that ignores a critical issue: {argument}",
        "On the contrary, {argument}"
    ],
    "Proponent": [
        "I see your point, since: {argument}",
        "One reason to back your claim is: {argument}",
        "Here’s another strong case in favor: {argument}",
        "That actually strengthens your position: {argument}"
    ],
}
COACH_NEUTRAL = [
    "Consider this related argument: {argument}",
    "You might find this point insightful: {argument}",
    "Here's a thought that may strengthen or challenge your view: {argument}",
    "This perspective might help you reflect further: {argument}",
    "Think about this argument: {argument}"
]
FALLBACK = "Consider this perspective: {argument}"

# ── Session State ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "used_opp" not in st.session_state:
    st.session_state.used_opp = set()
if "used_prop" not in st.session_state:
    st.session_state.used_prop = set()
if "coach_used" not in st.session_state:
    st.session_state.coach_used = set()
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# ── Render Chat ────────────────────────────────────────────────────────────────
def render_chat():
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='text-align:right; background:#e1f5fe; padding:8px;"
                f" border-radius:8px; margin:4px; color:black;'>"
                f"<strong>You:</strong> {msg['text']}</div>",
                unsafe_allow_html=True
            )
        else:
            bubbles = msg["content"] if isinstance(msg["content"], list) else [msg["content"]]
            for bubble in bubbles:
                st.markdown(
                    f"<div style='text-align:left; background:#f1f8e9; padding:8px;"
                    f" border-radius:8px; margin:4px; color:black;'>{bubble}</div>",
                    unsafe_allow_html=True
                )

st.title("Logarg")
render_chat()
st.markdown("---")

# ── Input Box ─────────────────────────────────────────────────────────────────
user_input = st.text_input(
    "Your statement:", key="input_box",
    value="" if st.session_state.clear_input else st.session_state.get('input_box', "")
)
if st.session_state.clear_input:
    st.session_state.clear_input = False

# ── Respond Handler ─────────────────────────────────────────────────────────────
if st.button("Respond"):
    if not user_input.strip():
        st.error("Please enter a statement.")
    else:
        st.session_state.history.append({"role": "user", "text": user_input})
        if st.session_state.get("last_input") != user_input:
            st.session_state.used_opp.clear()
            st.session_state.used_prop.clear()
            st.session_state.coach_used.clear()
            st.session_state.last_input = user_input

        arg_texts    = filtered_db["argument"].tolist()
        orig_indices = filtered_db["index"].tolist()

        enc = rel_tok([user_input] * len(arg_texts),
                      arg_texts,
                      padding=True,
                      truncation=True,
                      return_tensors="pt")
        with torch.no_grad():
            logits = rel_mod(**enc).logits
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        recs = []
        for pos, (arg, idx) in enumerate(zip(arg_texts, orig_indices)):
            lid = int(np.argmax(probs[pos]))
            recs.append({
                "idx":      idx,
                "argument": arg,
                "relation": REL_LABELS[lid],
                "prob":     float(probs[pos][lid])
            })
        df = pd.DataFrame(recs)

        if mode == "Opponent":
            df = df[~df["idx"].isin(st.session_state.used_opp)]
        elif mode == "Proponent":
            df = df[~df["idx"].isin(st.session_state.used_prop)]
        else:
            df = df[~df["idx"].isin(st.session_state.coach_used)]

        if mode == "Opponent":
            df = df[df["relation"] == "attack"]
        elif mode == "Proponent":
            df = df[df["relation"] == "support"]
        elif mode == "Debating Coach":
            df = df[df["relation"].isin(["support", "attack"])]

        if df.empty:
            st.session_state.history.append({
                "role":    "assistant",
                "content": "I’m sorry, I couldn’t find any relevant arguments."
            })
        else:
            if mode in ["Opponent", "Proponent"]:
                rec = df.sort_values("prob", ascending=False).iloc[0]
                text = f"{rec['argument']} (Confidence: {rec['prob']:.2f})"
                if mode == "Opponent":
                    st.session_state.used_opp.add(rec["idx"])
                else:
                    st.session_state.used_prop.add(rec["idx"])
                st.session_state.history.append({
                    "role":    "assistant",
                    "content": [text]
                })
            else:
                top5 = df.sort_values("prob", ascending=False).head(5).to_dict("records")
                msgs = []
                for rec in top5:
                    st.session_state.coach_used.add(rec["idx"])
                    label     = rec["relation"]
                    formatted = f"[{label.upper()} | Confidence: {rec['prob']:.2f}] {rec['argument']}"
                    msgs.append(
                        random.choice(COACH_NEUTRAL).format(argument=formatted)
                    )
                st.session_state.history.append({
                    "role":    "assistant",
                    "content": msgs
                })

        st.session_state.clear_input = True
        st.experimental_rerun()