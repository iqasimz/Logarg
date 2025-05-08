import streamlit as st
import pandas as pd
import torch
import time
import json
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(layout="centered")

# ── Load Relation Model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_relation_model():
    rel_repo = "iqasimz/logarg-relationtagger"
    tok = AutoTokenizer.from_pretrained(rel_repo)
    mod = AutoModelForSequenceClassification.from_pretrained(rel_repo).eval()
    return tok, mod

rel_tok, rel_mod = load_relation_model()
REL_LABELS = ["attack", "support", "none"]

# ── Load Databank ───────────────────────────────────────────────────────────────
@st.cache_data
def load_databank(path="databank.jsonl"):
    return pd.read_json(path, lines=True)

db = load_databank()

@st.cache_data
def load_tfidf(corpus):
    vectorizer = TfidfVectorizer().fit(corpus)
    matrix = vectorizer.transform(corpus)
    return vectorizer, matrix

vectorizer, arg_tfidf = load_tfidf(db["argument"].tolist())

# ── Session State ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "used" not in st.session_state:
    st.session_state.used = []
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# ── Sidebar: Mode & User Stance ─────────────────────────────────────────────────
mode = st.sidebar.selectbox("Mode", ["Proponent", "Opponent", "Debating Coach"])
user_stance = st.sidebar.selectbox("Your stance on nuclear energy", ["Pro", "Con"]).lower()

# ── Chat Rendering ──────────────────────────────────────────────────────────────
def render_chat():
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='text-align:right; background:#e1f5fe; padding:8px; "
                f"border-radius:8px; margin:4px; color:black;'>"
                f"<strong>You:</strong> {msg['text']}</div>",
                unsafe_allow_html=True
            )
        else:
            with st.spinner("🤖 Assistant is typing..."):
                pass
            if msg["mode"] in ["Opponent", "Proponent"]:
                content = f"**{msg['relation'].title()} Argument:** {msg['argument']}"
            else:
                lines = "".join(
                    f"<li>({r['relation']}/{r['score']:.3f}) {r['argument']}</li>"
                    for r in msg["arguments"]
                )
                content = f"**Top 20 Arguments (All Relations):**<ul>{lines}</ul>"
            st.markdown(
                f"<div style='text-align:left; background:#f1f8e9; padding:8px; "
                f"border-radius:8px; margin:4px; color:black;'>"
                f"{content}</div>",
                unsafe_allow_html=True
            )

# ── Main UI ─────────────────────────────────────────────────────────────────────
st.title("Logarg Debate Assistant")
render_chat()
st.markdown("---")

initial_value = "" if st.session_state.clear_input else st.session_state.get("input_box", "")
user_input = st.text_input("Your statement:", value=initial_value, key="input_box")
if st.session_state.clear_input:
    st.session_state.clear_input = False

if st.button("Respond", key="respond_button"):
    if not user_input.strip():
        st.error("Please enter a statement.")
    else:
        st.session_state.history.append({"role": "user", "text": user_input})

        # 1) Relation classification
        texts_user = [user_input] * len(db)
        enc = rel_tok(texts_user, db["argument"].tolist(),
                      padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(rel_mod(**enc).logits, dim=1).cpu().tolist()

        # 2) TF-IDF similarity only (cached)
        user_tfidf = vectorizer.transform([user_input])
        sims = cosine_similarity(arg_tfidf, user_tfidf).flatten()

        used_set = set(st.session_state.used)

        recs = []
        for idx, row in db.iterrows():
            if idx in used_set:
                continue
            prob = probs[idx]
            lid = int(np.argmax(prob))
            label = REL_LABELS[lid]
            if user_stance == "con":
                if label == "support":
                    label = "attack"
                elif label == "attack":
                    label = "support"
            sim = float(sims[idx])
            recs.append({
                "idx": idx,
                "argument": row["argument"],
                "relation": label,
                "score": sim
            })

        df = pd.DataFrame(recs)

        # 3) Filtering
        if mode == "Opponent":
            df_filtered = df[df["relation"] == "attack"].copy()
        elif mode == "Proponent":
            df_filtered = df[df["relation"] == "support"].copy()
        else:
            df_filtered = df.copy()

        if df_filtered.empty:
            st.warning("No matching arguments found.")
        else:
            df_sorted = df_filtered.sort_values("score", ascending=False)

            if mode in ["Opponent", "Proponent"]:
                best = df_sorted.iloc[0]
                st.session_state.used.append(int(best["idx"]))
                st.session_state.history.append({
                    "role": "assistant",
                    "mode": mode,
                    "relation": best["relation"],
                    "argument": best["argument"]
                })
            else:
                topn = df_sorted.head(5).to_dict("records")
                for r in topn:
                    st.session_state.used.append(int(r["idx"]))
                st.session_state.history.append({
                    "role": "assistant",
                    "mode": mode,
                    "arguments": topn
                })

        st.session_state.clear_input = True
        st.experimental_rerun()
