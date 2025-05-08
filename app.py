import streamlit as st
import pandas as pd
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(layout="centered")

# ── Load Models ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rel_repo = "iqasimz/logarg-relationtagger"
    rel_tok  = AutoTokenizer.from_pretrained(rel_repo)
    rel_mod  = AutoModelForSequenceClassification.from_pretrained(rel_repo).eval()
    return rel_tok, rel_mod

rel_tok, rel_mod = load_models()
REL_LABELS = ["attack", "support", "none"]

# ── Load Databank ───────────────────────────────────────────────────────────────
@st.cache_data
def load_databank(path="databank.jsonl"):
    return pd.read_json(path, lines=True)

db = load_databank()

# ── Session State ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # chat messages
if "used" not in st.session_state:
    st.session_state.used = set()   # used argument indices
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# ── Sidebar: Mode & User Stance ─────────────────────────────────────────────────
mode         = st.sidebar.selectbox("Mode", ["Proponent", "Opponent", "Debating Coach"])
user_stance  = st.sidebar.selectbox("Your stance on nuclear energy", ["Pro", "Con"]).lower()

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
                time.sleep(0.5)
            if msg["mode"] in ["Opponent", "Proponent"]:
                content = f"**{msg['relation'].title()} Argument:** {msg['argument']}"
            else:
                lines = "".join(
                    f"<li>({r['relation']}/{r['probability']:.3f}) {r['argument']}</li>"
                    for r in msg["arguments"]
                )
                content = f"**Top Suggestions:**<ul>{lines}</ul>"
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

# Determine initial input box value
initial_value = "" if st.session_state.clear_input else st.session_state.get("input_box", "")
user_input = st.text_input("Your statement:", value=initial_value, key="input_box")

# After displaying, reset clear_input flag
if st.session_state.clear_input:
    st.session_state.clear_input = False

if st.button("Respond", key="respond_button"):
    if not user_input.strip():
        st.error("Please enter a statement.")
    else:
        # Record user message
        st.session_state.history.append({"role": "user", "text": user_input})

        # Batch inference against databank
        texts_user = [user_input] * len(db)
        enc = rel_tok(texts_user, db["argument"].tolist(),
                      padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(rel_mod(**enc).logits, dim=1).cpu().tolist()

        # Collect and flip labels if needed
        recs = []
        for idx, (row, prob) in enumerate(zip(db.to_dict("records"), probs)):
            if idx in st.session_state.used:
                continue
            lid   = int(torch.argmax(torch.tensor(prob)))
            label = REL_LABELS[lid]
            # Flip for "Con" stance
            if user_stance == "con":
                if label == "support":
                    label = "attack"
                elif label == "attack":
                    label = "support"
            recs.append({
                "idx": idx,
                "argument": row["argument"],
                "relation": label,
                "probability": prob[lid]
            })

        df = pd.DataFrame(recs)
        # Mode filtering
        if mode == "Opponent":
            df = df[df["relation"] == "attack"]
        elif mode == "Proponent":
            df = df[df["relation"] == "support"]
        if df.empty:
            st.warning("No matching arguments found.")
        else:
            df = df.sort_values("probability", ascending=False)
            if mode in ["Opponent", "Proponent"]:
                best = df.iloc[0]
                st.session_state.used.add(int(best["idx"]))
                st.session_state.history.append({
                    "role":     "assistant",
                    "mode":     mode,
                    "relation": best["relation"],
                    "argument": best["argument"]
                })
            else:
                topn = df.head(20).to_dict("records")
                for r in topn:
                    st.session_state.used.add(int(r["idx"]))
                st.session_state.history.append({
                    "role":      "assistant",
                    "mode":      mode,
                    "arguments": topn
                })

        # Signal to clear input next render
        st.session_state.clear_input = True
        st.experimental_rerun()