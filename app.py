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

# Use a dedicated SBERT model for retrieval
RETRIEVAL_MODEL = SBERT_MODEL

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
def build_index(texts, emb_model):
    # Compute SBERT embeddings for retrieval
    emb_matrix = emb_model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True
    ).astype('float32')
    faiss.normalize_L2(emb_matrix)
    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)
    return index, emb_matrix

arg_texts = filtered_db['argument'].tolist()
index, arg_embs = build_index(arg_texts, RETRIEVAL_MODEL)

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

        # Reset all “used” sets on a new user statement
        if st.session_state.get("last_input") != user_input:
            st.session_state.used_opp.clear()
            st.session_state.used_prop.clear()
            st.session_state.coach_used.clear()
            st.session_state.last_input = user_input

        # Detect true syntactic negation
        has_syntactic_neg = any(tok.dep_ == "neg" for tok in nlp(user_input))
        neg_flag = has_syntactic_neg

        # Detect strong negative sentiment for override
        sent_scores = VADER.polarity_scores(user_input)
        neg_sent = sent_scores["compound"] < -0.3

        # 1) Compute user embedding
        enc = rel_tok([user_input], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            user_emb = rel_mod.base_model(**enc).last_hidden_state.mean(1).cpu().numpy().astype('float32')
        faiss.normalize_L2(user_emb)

        # 2) Retrieve candidates
        k = 150
        if mode in ["Opponent", "Debating Coach"]:
            # Narrow down to top-k candidates before classification
            D_opp, I_opp = index.search(user_emb, k)
            valid_idxs = [i for i in I_opp[0] if i >= 0]
            batch_args = [arg_texts[i] for i in valid_idxs]
            orig_indices = [filtered_db['index'].tolist()[i] for i in valid_idxs]
        else:  # Proponent mode
            D, I = index.search(user_emb, k)
            # Remove invalid (-1) indices from FAISS output
            faiss_scores = D[0].tolist()
            faiss_idxs   = I[0].tolist()
            valid_pairs = [(score, idx) for score, idx in zip(faiss_scores, faiss_idxs) if idx >= 0]
            if valid_pairs:
                scores_list, idx_list = zip(*valid_pairs)
                D = [list(scores_list)]
                I = [list(idx_list)]
                batch_args = [arg_texts[i] for i in idx_list]
            else:
                D = [[]]
                I = [[]]
                batch_args = []

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
                row_probs = probs[pos]
                lid = int(np.argmax(row_probs))
                rel = REL_LABELS[lid]
                if neg_flag and rel in ("support","attack"):
                    rel = "attack" if rel=="support" else "support"
                # sentiment-based override for none
                if rel == "none" and neg_sent:
                    rel = "attack"
                recs.append({
                    'idx': orig_idx,
                    'argument': arg,
                    'relation': rel,
                    'score': 0.0,
                    'prob': float(row_probs[lid]),
                    'attack_prob': float(row_probs[0])  # track probability for attack label
                })
        elif mode == "Proponent":
            for pos, (arg, orig_idx) in enumerate(zip(batch_args, orig_indices)):
                lid = int(np.argmax(probs[pos]))
                rel = REL_LABELS[lid]
                if neg_flag and rel in ("support","attack"):
                    rel = "attack" if rel=="support" else "support"
                # sentiment-based override for none
                if rel == "none" and neg_sent:
                    rel = "attack"
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
                # sentiment-based override for none
                if rel == "none" and neg_sent:
                    rel = "attack"
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
            if mode == "Opponent" and recs:
                # Fallback: pick the argument with highest predicted attack probability
                best = max(recs, key=lambda r: r.get('attack_prob', 0.0))
                msg = f"{best['argument']} (Attack confidence: {best['attack_prob']:.2f})"
                st.session_state.history.append({"role": "assistant", "content": [msg]})
            else:
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

        # 9) Reset input
        st.session_state.clear_input = True