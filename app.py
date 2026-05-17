"""
app.py

Improved Streamlit UI and prediction helper for the GRU spam detector.
Adds stronger text cleaning, batch CSV upload, token highlighting and
safer error handling when model/tokenizer files are missing.
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
import pandas as pd
import html
import nltk
from nltk.stem import WordNetLemmatizer

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📩",
    layout="wide",
)

# ------------------------------
# Ensure NLTK data
# ------------------------------
try:
    nltk.data.find("corpora/wordnet")
except Exception:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

# ------------------------------
# Custom CSS (slim)
# ------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #071029; }
    .title { text-align:center; font-size:40px; color:#f8fafc; font-weight:700; }
    .subtitle { text-align:center; color:#cbd5e1; margin-bottom:18px }
    .result-box { padding:18px; border-radius:12px; text-align:center; font-size:20px; font-weight:700 }
    .spam { background:#7f1d1d; color:#fecaca }
    .ham { background:#14532d; color:#bbf7d0 }
    .stTextArea textarea { background:#0b1220; color:#e6eef8; border-radius:10px }
    .footer { text-align:center; color:#9fb0c8 }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Load artifacts safely
# ------------------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("gru_model.keras")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open("tokenizer.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load tokenizer: {e}")
        return None

@st.cache_resource
def load_config():
    try:
        with open("config.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Using default config because: {e}")
        return {"max_length": 100}

@st.cache_resource
def load_label_mapping():
    try:
        with open("label_mapping.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        # sensible default
        return {0: "Ham", 1: "Spam"}

model = load_model()
tokenizer = load_tokenizer()
config = load_config()
label_mapping = load_label_mapping()

MAX_LENGTH = int(config.get("max_length", 100))

# ------------------------------
# Text Cleaning & preprocessing
# ------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # unescape html entities
    text = html.unescape(text)
    # remove html tags
    text = re.sub(r"<.*?>", " ", text)
    # remove urls and emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    # keep only letters and numbers and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    # simple lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in text.split()]
    return " ".join(tokens)

# ------------------------------
# Prediction Function
# ------------------------------
def predict_spam(message: str):
    if model is None or tokenizer is None:
        return "Unavailable", 0.0

    cleaned = clean_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post")

    try:
        prediction = float(model.predict(padded, verbose=0)[0][0])
    except Exception:
        prediction = 0.0

    predicted_class = 1 if prediction > 0.5 else 0
    label = label_mapping.get(predicted_class, "Spam" if predicted_class == 1 else "Ham")
    return label, prediction

def extract_tokens(message: str):
    cleaned = clean_text(message)
    seq = tokenizer.texts_to_sequences([cleaned])[0] if tokenizer else []
    tokens = []
    if tokenizer and hasattr(tokenizer, "index_word"):
        for idx in seq:
            word = tokenizer.index_word.get(idx)
            if word:
                tokens.append(word)
    return tokens

# ------------------------------
# UI
# ------------------------------
st.markdown('<div class="title">📩 Spam Message Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">GRU model — improved cleaning, batch mode, and clearer UI</div>', unsafe_allow_html=True)

# layout: main + sidebar
left, right = st.columns([3, 1])

with left:
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    user_input = st.text_area("Enter your message below:", value=st.session_state.user_input, height=180, placeholder="Type your SMS or Email here...")

    # sample quick-fill buttons
    def _set_example_spam():
        st.session_state.user_input = "Congratulations! You have won a $1000 gift card. Click the link now to claim."

    def _set_example_ham():
        st.session_state.user_input = "Hey, are we still on for the meeting tomorrow at 10am?"

    def _clear_input():
        st.session_state.user_input = ""

    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("Example: Spam", on_click=_set_example_spam)
    with c2:
        st.button("Example: Ham", on_click=_set_example_ham)
    with c3:
        st.button("Clear", on_click=_clear_input)

    predict_button = st.button("🔍 Check Message")

    if predict_button:
        if not user_input or user_input.strip() == "":
            st.warning("Please enter a message to analyze.")
        else:
            with st.spinner("Analyzing message..."):
                label, prob = predict_spam(user_input)
            spam_pct = round(prob * 100, 2)
            ham_pct = round((1 - prob) * 100, 2)

            if label.lower() == "spam":
                st.markdown(f"<div class=\"result-box spam\">🚨 SPAM • {spam_pct}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class=\"result-box ham\">✅ HAM • {ham_pct}%</div>", unsafe_allow_html=True)

            # show both bars
            st.markdown("### Prediction Probabilities")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.write("Spam")
                st.progress(int(spam_pct))
            with pcol2:
                st.write("Ham")
                st.progress(int(ham_pct))

            # tokens
            tokens = extract_tokens(user_input)
            if tokens:
                st.write("**Detected tokens (from tokenizer):**", ", ".join(tokens[:30]))

            with st.expander("📊 View Technical Details"):
                st.write(f"Raw Prediction Score (spam prob): {prob:.4f}")
                st.write(f"Predicted Label: {label}")
                st.write(f"Model Loaded: {'Yes' if model is not None else 'No'}")
                st.write(f"Tokenizer Loaded: {'Yes' if tokenizer is not None else 'No'}")
                st.write(f"Max Sequence Length: {MAX_LENGTH}")

    # Batch upload
    st.markdown("---")
    st.write("### Batch prediction (CSV)")
    uploaded = st.file_uploader("Upload CSV with a text column", type=["csv"] )
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Columns:", list(df.columns))
            text_col = st.selectbox("Select text column", options=df.columns)
            if st.button("Run batch prediction"):
                with st.spinner("Running batch predictions..."):
                    df["cleaned_text"] = df[text_col].astype(str).apply(clean_text)
                    sequences = tokenizer.texts_to_sequences(df["cleaned_text"].tolist())
                    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post")
                    preds = model.predict(padded, verbose=0).reshape(-1)
                    df["spam_prob"] = preds
                    df["predicted_label"] = df["spam_prob"].apply(lambda p: label_mapping.get(1 if p>0.5 else 0))
                st.write(df[[text_col, "predicted_label", "spam_prob"]].head(50))
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to run batch prediction: {e}")

with right:
    st.header("ℹ About")
    st.write(
        "This app uses a GRU model to classify messages as Spam or Ham. Improvements include stronger text cleaning, token highlights and a batch CSV mode."
    )
    st.markdown("---")
    st.subheader("Tech Stack")
    st.write("- TensorFlow / Keras")
    st.write("- Streamlit")
    st.write("- NLTK (lemmatization)")
    st.markdown("---")
    st.subheader("Examples")
    st.info("Spam example: 'You won a prize! Click here to claim.'")
    st.success("Ham example: 'Can we reschedule our meeting to Monday?'")

st.markdown('<div class="footer">Made with ❤️ using Streamlit & TensorFlow</div>', unsafe_allow_html=True)
