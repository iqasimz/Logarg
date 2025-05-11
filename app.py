import os
# Must come before any imports of torch/faiss/etc.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import torch
import numpy as np
import faiss
import random
import time

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load spaCy and VADER
nlp = spacy.load("en_core_web_sm")
VADER = SentimentIntensityAnalyzer()

# SBERT for topic-filtering
SBERT_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')
TOPIC_THRESHOLD = 0.55  # similarity threshold for “none” filtering

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(layout="centered")

# ── Load Databank ─────────────────────────────────────────────────────────────
@st.cache_data
def load_databank(path="databank.jsonl"):
    return pd.read_json(path, lines=True)

db = load_databank()

# ── Sidebar: Topic, Mode ───────────────────────────────────────────────────────
topics = sorted(db['topic'].unique())
selected_topic = st.sidebar.selectbox("Select debate topic:", topics)
filtered_db = db[db['topic'] == selected_topic].reset_index(drop=False)
mode = st.sidebar.selectbox("Mode", ["Proponent", "Opponent", "Debating Coach"])

# ── Load Relation Model & Tokenizer ────────────────────────────────────────────
@st.cache_resource
def load_relation_model(repo="iqasimz/logarg-relationtagger"):
    tok = AutoTokenizer.from_pretrained(repo)
    mod = AutoModelForSequenceClassification.from_pretrained(repo)
    mod.eval()
    return tok, mod

rel_tok, rel_mod = load_relation_model()
REL_LABELS = ["attack", "support", "none"]

# ── Precompute Argument Embeddings & Build FAISS Index ─────────────────────────
@st.cache_resource
def build_index(texts, _tok, _model, emb_dim=768):
    all_embs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = _tok(batch, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outs = _model.base_model(**enc).last_hidden_state.mean(1)
        all_embs.append(outs.cpu().numpy())
    emb_matrix = np.vstack(all_embs).astype('float32')
    faiss.normalize_L2(emb_matrix)
    index = faiss.IndexFlatIP(emb_dim)
    index.add(emb_matrix)
    return index, emb_matrix

arg_texts = filtered_db['argument'].tolist()
index, arg_embs = build_index(arg_texts, rel_tok, rel_mod)

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
COACH_SUPPORT = [
    "Here’s how one might support your claim: {support_argument}",
    "Good start. You could strengthen your case with: {support_argument}",
    "Another advantage is {support_argument}",
    "You might also consider {support_argument}",
    "Additionally, {support_argument} is worth mentioning",
    "An added benefit is {support_argument}"
]
COACH_ATTACK = [
    "But on the flip side, some argue: {attack_argument}",
    "To anticipate opposition, consider: {attack_argument}",
    "However, {attack_argument}",
    "A significant drawback is {attack_argument}",
    "Critically, {attack_argument}",
    "One must note {attack_argument}"
]
FALLBACK = "Consider this perspective: {argument}"

# ── Session State ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "used" not in st.session_state:
    st.session_state.used = set()
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

st.title("Logarg Debate Assistant")
render_chat()
st.markdown("---")

# ── Input Box ─────────────────────────────────────────────────────────────────
user_input = st.text_input(
    "Your statement:", key="input_box",
    value="" if st.session_state.clear_input else st.session_state.get('input_box', '')
)
if st.session_state.clear_input:
    st.session_state.clear_input = False

# ── Respond Handler ─────────────────────────────────────────────────────────────
if st.button("Respond"):
    if not user_input.strip():
        st.error("Please enter a statement.")
    else:
        # Append user message
        st.session_state.history.append({"role": "user", "text": user_input})

        # Reset Opponent pool on new claim
        if st.session_state.get("last_input") != user_input:
            st.session_state.used_opp.clear()
            st.session_state.last_input = user_input

        # Detect negation + sentiment
        has_syntactic_neg = any(tok.dep_ == "neg" for tok in nlp(user_input))
        sent_scores = VADER.polarity_scores(user_input)
        has_negative_sent = sent_scores["compound"] < -0.05
        neg_flag = has_syntactic_neg or has_negative_sent

        # 1) Compute user embedding
        enc = rel_tok([user_input], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            user_emb = rel_mod.base_model(**enc).last_hidden_state.mean(1).cpu().numpy().astype('float32')
        faiss.normalize_L2(user_emb)

        # 2) Retrieve candidates
        k = 150
        if mode in ["Opponent", "Debating Coach"]:
            batch_args = arg_texts
        else:
            D, I = index.search(user_emb, k)
            batch_args = [arg_texts[i] for i in I[0]]

        # Track original indices
        if mode in ["Opponent", "Debating Coach"]:
            orig_indices = filtered_db['index'].tolist()
        else:
            orig_indices = [int(filtered_db.loc[i, 'index']) for i in I[0]]

        # 3) Stage 1: SBERT topic filter (skip for Opponent)
        if mode == "Opponent":
            keep_idx = list(range(len(batch_args)))
        else:
            user_emb_sbert = SBERT_MODEL.encode([user_input], convert_to_tensor=True)
            arg_embs_sbert = SBERT_MODEL.encode(batch_args, convert_to_tensor=True)
            cos_scores = F.cosine_similarity(
                arg_embs_sbert,
                user_emb_sbert.expand_as(arg_embs_sbert),
                dim=1
            )
            keep_idx = [i for i, s in enumerate(cos_scores) if s.item() > TOPIC_THRESHOLD]
            if not keep_idx:
                recs = [{
                    'idx': None,
                    'argument': arg,
                    'relation': 'none',
                    'score': 0.0,
                    'prob': 1.0
                } for arg in batch_args[:3]]
                msgs = [f"{r['argument']} (Confidence: {r['prob']:.2f})" for r in recs]
                st.session_state.history.append({"role": "assistant", "content": msgs})
                st.session_state.clear_input = True
                st.experimental_rerun()

        # Filter candidates for stage 2
        batch_args = [batch_args[i] for i in keep_idx]
        orig_indices = [orig_indices[i] for i in keep_idx]
        if mode == "Proponent":
            D = [[D[0][i] for i in keep_idx]]
            I = [[I[0][i] for i in keep_idx]]

        # 4) Stage 2: Relation classification
        batch_texts = [user_input] * len(batch_args)
        enc2 = rel_tok(batch_args, batch_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = rel_mod(**enc2).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # 5) Build recs
        recs = []
        if mode == "Opponent":
            for pos, (arg, orig_idx) in enumerate(zip(batch_args, orig_indices)):
                lid = int(np.argmax(probs[pos]))
                rel = REL_LABELS[lid]
                if neg_flag and rel in ("support","attack"):
                    rel = "attack" if rel=="support" else "support"
                recs.append({
                    'idx': orig_idx,
                    'argument': arg,
                    'relation': rel,
                    'score': 0.0,
                    'prob': float(probs[pos][lid])
                })
        elif mode == "Proponent":
            for pos, (arg, orig_idx) in enumerate(zip(batch_args, orig_indices)):
                lid = int(np.argmax(probs[pos]))
                rel = REL_LABELS[lid]
                if neg_flag and rel in ("support","attack"):
                    rel = "attack" if rel=="support" else "support"
                score = float(D[0][pos]) if 'D' in locals() else 0.0
                recs.append({
                    'idx': orig_idx,
                    'argument': arg,
                    'relation': rel,
                    'score': score,
                    'prob': float(probs[pos][lid])
                })
        else:  # Debating Coach
            for pos, (arg, orig_idx) in enumerate(zip(batch_args, orig_indices)):
                lid = int(np.argmax(probs[pos]))
                rel = REL_LABELS[lid]
                if neg_flag and rel in ("support","attack"):
                    rel = "attack" if rel=="support" else "support"
                recs.append({
                    'idx': orig_idx,
                    'argument': arg,
                    'relation': rel,
                    'score': 0.0,
                    'prob': float(probs[pos][lid])
                })

        df = pd.DataFrame(recs)

        # 6) Filter out used per mode
        if mode == 'Opponent':
            df = df[~df['idx'].isin(st.session_state.used_opp)]
        elif mode == 'Proponent':
            df = df[~df['idx'].isin(st.session_state.used_prop)]
        else:
            df = df[~df['idx'].isin(st.session_state.coach_used)]

        # 7) Filter by relation
        if mode == 'Opponent':
            df = df[df['relation'] == 'attack']
        elif mode == 'Proponent':
            df = df[df['relation'] == 'support']

        # 8) Display
        if df.empty:
            st.session_state.history.append({
                "role": "assistant",
                "content": "I’m sorry, I couldn’t find any relevant arguments."
            })
        else:
            if mode in ['Opponent', 'Proponent']:
                top1 = df.head(1).to_dict('records')
                if top1:
                    rec = top1[0]
                    prob = rec.get('prob', 0.0)
                    msg = f"{rec['argument']} (Confidence: {prob:.2f})"
                    if mode == 'Opponent':
                        st.session_state.used_opp.add(rec['idx'])
                    else:
                        st.session_state.used_prop.add(rec['idx'])
                    st.session_state.history.append({"role": "assistant", "content": [msg]})
            else:
                top5 = df.head(10).to_dict('records')
                msgs = []
                for rec in top5:
                    st.session_state.coach_used.add(rec['idx'])
                    label = rec['relation']
                    formatted = f"[{label.upper()}] {rec['argument']}"
                    if label == 'support':
                        msgs.append(random.choice(COACH_SUPPORT).format(support_argument=formatted))
                    elif label == 'attack':
                        msgs.append(random.choice(COACH_ATTACK).format(attack_argument=formatted))
                    else:
                        msgs.append(FALLBACK.format(argument=formatted))
                st.session_state.history.append({"role": "assistant", "content": msgs})

        # 9) Reset input & rerun
        st.session_state.clear_input = True
        st.experimental_rerun()