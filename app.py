import os
# Must come before any imports of torch/faiss/etc.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
# ...
import streamlit as st
import pandas as pd
import torch
import numpy as np
import faiss
import random
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(layout="centered")

# ── Load Databank ─────────────────────────────────────────────────────────────
@st.cache_data
def load_databank(path="databank.jsonl"):
    return pd.read_json(path, lines=True)

db = load_databank()

# ── Sidebar: Topic, Mode, Stance ───────────────────────────────────────────────
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
# Note: Leading underscores on tok and model tell Streamlit not to hash them
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
        st.session_state.history.append({"role": "user", "text": user_input})

        # 1) Compute user embedding
        enc = rel_tok([user_input], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            user_emb = rel_mod.base_model(**enc).last_hidden_state.mean(1).cpu().numpy().astype('float32')
        faiss.normalize_L2(user_emb)

        # 2) Retrieve top-k or use all arguments based on mode
        k = 150
        if mode in ["Opponent", "Debating Coach"]:
            # Use full set for Opponent and Coach to avoid FAISS bias
            batch_args = arg_texts
        else:  # Proponent mode
            D, I = index.search(user_emb, k)
            batch_args = [arg_texts[i] for i in I[0]]
        # 3) Batch relation classification on selected arguments
        batch_texts = [user_input] * len(batch_args)
        enc2 = rel_tok(batch_args, batch_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = rel_mod(**enc2).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # 3) Build recs list per mode
        recs = []
        if mode == "Opponent":
            # Use all arguments, no scores
            for pos, arg in enumerate(arg_texts):
                orig_idx = int(filtered_db.loc[pos, 'index'])
                lid = int(np.argmax(probs[pos]))
                recs.append({
                    'idx': orig_idx,
                    'argument': arg,
                    'relation': REL_LABELS[lid],
                    'score': 0.0
                })
        elif mode == "Proponent":
            # Use top-k from FAISS
            D, I = index.search(user_emb, k)
            for pos, db_idx in enumerate(I[0]):
                orig_idx = int(filtered_db.loc[db_idx, 'index'])
                lid = int(np.argmax(probs[pos]))
                recs.append({
                    'idx': orig_idx,
                    'argument': arg_texts[db_idx],
                    'relation': REL_LABELS[lid],
                    'score': float(D[0][pos])
                })
        else:  # Debating Coach
            # Use all arguments, no scores
            for pos, arg in enumerate(arg_texts):
                orig_idx = int(filtered_db.loc[pos, 'index'])
                lid = int(np.argmax(probs[pos]))
                recs.append({
                    'idx': orig_idx,
                    'argument': arg,
                    'relation': REL_LABELS[lid],
                    'score': 0.0
                })
        df = pd.DataFrame(recs)

        # Filter by mode and used arguments with updated logic
        if mode in ['Opponent', 'Proponent']:
            # Arguments used in Attack or Support mode cannot be reused in those modes but can be reused in Coach mode
            df = df[~df['idx'].isin(st.session_state.used)]
        else:
            # For Coach mode, filter out only those already shown
            df = df[~df['idx'].isin(st.session_state.coach_used)]

        if mode == 'Opponent':
            df = df[df['relation'] == 'attack']
        elif mode == 'Proponent':
            df = df[df['relation'] == 'support']

        if df.empty:
            st.session_state.history.append({
                "role": "assistant",
                "content": "I’m sorry, I couldn’t find any relevant arguments."
            })
        else:
            if mode in ['Opponent', 'Proponent']:
                # Show top 3 arguments with probabilities
                top3 = df.head(3).to_dict('records')
                msgs = []
                for rec in top3:
                    prob = max(probs[0][int(rec['relation'] == 'attack')], probs[0][int(rec['relation'] == 'support')])
                    msgs.append(f"{rec['argument']} (Confidence: {prob:.2f})")
                    st.session_state.used.add(rec['idx'])
                st.session_state.history.append({"role": "assistant", "content": msgs})
            else:
                # Coach mode: show up to 5 fresh arguments and mark only those displayed
                top5 = df.head(5).to_dict('records')
                msgs = []
                for rec in top5:
                    # Mark only the ones we display
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

        # 5) Reset input & rerun
        st.session_state.clear_input = True
        st.experimental_rerun()