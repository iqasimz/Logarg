import streamlit as st
import pandas as pd
import torch
import numpy as np
import random
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(layout="centered")

# ── Load Databank ───────────────────────────────────────────────────────────────
@st.cache_data
def load_databank(path="databank.jsonl"):
    return pd.read_json(path, lines=True)

db = load_databank()

# ── Topic Selection ─────────────────────────────────────────────────────────────
topics = sorted(db['topic'].unique())
selected_topic = st.sidebar.selectbox("Select debate topic:", topics)
filtered_db = db[db['topic'] == selected_topic].reset_index(drop=False)

# ── Load Relation Model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_relation_model():
    rel_repo = "iqasimz/logarg-relationtagger"
    tok = AutoTokenizer.from_pretrained(rel_repo)
    mod = AutoModelForSequenceClassification.from_pretrained(rel_repo).eval()
    return tok, mod

rel_tok, rel_mod = load_relation_model()
REL_LABELS = ["attack", "support", "none"]

# ── Precompute TF-IDF ───────────────────────────────────────────────────────────
@st.cache_data
def load_tfidf(corpus):
    vectorizer = TfidfVectorizer().fit(corpus)
    matrix     = vectorizer.transform(corpus)
    return vectorizer, matrix

vectorizer, arg_tfidf = load_tfidf(filtered_db["argument"].tolist())

# ── Templates for Each Mode ─────────────────────────────────────────────────────
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
COACH_SUPPORT_TEMPLATES = [
    "Here’s how one might support your claim: {support_argument}",
    "Good start. You could strengthen your case with: {support_argument}",
    "Another advantage is {support_argument}",
    "You might also consider {support_argument}",
    "Additionally, {support_argument} is worth mentioning",
    "An added benefit is {support_argument}"
]
COACH_ATTACK_TEMPLATES = [
    "But on the flip side, some argue: {attack_argument}",
    "To anticipate opposition, consider: {attack_argument}",
    "However, {attack_argument}",
    "A significant drawback is {attack_argument}",
    "Critically, {attack_argument}",
    "One must note {attack_argument}"
]
COACH_FALLBACK = "Consider this perspective: {argument}"

# ── Session State Initialization ────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "used" not in st.session_state:
    st.session_state.used = set()
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# ── Sidebar: Mode & User Stance ─────────────────────────────────────────────────
mode = st.sidebar.selectbox("Mode", ["Proponent", "Opponent", "Debating Coach"])
user_stance = st.sidebar.selectbox(f"Your stance on {selected_topic}", ["Pro", "Con"]).lower()

# ── Chat Rendering ──────────────────────────────────────────────────────────────
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
                with st.spinner("🤖 Assistant is typing..."):
                    time.sleep(0.4)
                st.markdown(
                    f"<div style='text-align:left; background:#f1f8e9; padding:8px;"
                    f" border-radius:8px; margin:4px; color:black;'>{bubble}</div>",
                    unsafe_allow_html=True
                )

# ── Main UI ─────────────────────────────────────────────────────────────────────
st.title("Logarg Debate Assistant")
render_chat()
st.markdown("---")

# Input box handling
initial_value = "" if st.session_state.clear_input else st.session_state.get("input_box", "")
user_input = st.text_input("Your statement:", value=initial_value, key="input_box")
if st.session_state.clear_input:
    st.session_state.clear_input = False

# ── Respond Handler ─────────────────────────────────────────────────────────────
if st.button("Respond"):
    if not user_input.strip():
        st.error("Please enter a statement.")
    else:
        # 1. Record user
        st.session_state.history.append({"role": "user", "text": user_input})

        # 2. Relation predictions
        texts = [user_input] * len(filtered_db)
        enc = rel_tok(texts, filtered_db["argument"].tolist(),
                      padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            rel_probs = torch.softmax(rel_mod(**enc).logits, dim=1).cpu().tolist()

        # 3. Similarity scores
        user_tfidf = vectorizer.transform([user_input])
        sims = cosine_similarity(arg_tfidf, user_tfidf).flatten()

        # 4. Collect & flip if needed
        recs = []
        for idx, row in filtered_db.iterrows():
            orig_idx = int(row["index"])
            if orig_idx in st.session_state.used:
                continue
            lid = int(np.argmax(rel_probs[idx]))
            label = REL_LABELS[lid]
            if user_stance == "con":
                if label == "support":
                    label = "attack"
                elif label == "attack":
                    label = "support"
            recs.append({
                "idx":      orig_idx,
                "argument": row["argument"],
                "relation": label,
                "score":    float(sims[idx])
            })

        df = pd.DataFrame(recs)

        # 5. Filter by mode
        if mode == "Opponent":
            df_filtered = df[df["relation"] == "attack"].copy()
        elif mode == "Proponent":
            df_filtered = df[df["relation"] == "support"].copy()
        else:
            df_filtered = df.copy()

        # 6. Generate assistant reply
        if df_filtered.empty:
            st.session_state.history.append({
                "role":    "assistant",
                "content": "I’m sorry, I couldn’t find any relevant arguments."
            })
        else:
            df_sorted = df_filtered.sort_values("score", ascending=False)
            if mode in ["Opponent", "Proponent"]:
                best = df_sorted.iloc[0]
                tpl = random.choice(TEMPLATES[mode])
                content = tpl.format(argument=best["argument"])
                st.session_state.used.add(best["idx"])
                st.session_state.history.append({
                    "role":    "assistant",
                    "content": content
                })
            else:
                # Debating Coach: top 5, no repeats
                top5 = df_sorted.head(5).to_dict("records")
                coach_msgs = []
                for rec in top5:
                    st.session_state.used.add(rec["idx"])
                    if rec["relation"] == "support":
                        tpl = random.choice(COACH_SUPPORT_TEMPLATES)
                        coach_msgs.append(tpl.format(support_argument=rec["argument"]))
                    elif rec["relation"] == "attack":
                        tpl = random.choice(COACH_ATTACK_TEMPLATES)
                        coach_msgs.append(tpl.format(attack_argument=rec["argument"]))
                    else:
                        coach_msgs.append(COACH_FALLBACK.format(argument=rec["argument"]))
                st.session_state.history.append({
                    "role":    "assistant",
                    "content": coach_msgs
                })

        # 7. Clear input & rerun
        st.session_state.clear_input = True
        st.experimental_rerun()