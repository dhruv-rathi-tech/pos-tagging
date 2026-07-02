import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import nltk
from nltk.tokenize import word_tokenize
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# ---------- Page config ----------
st.set_page_config(page_title="POS Tagger", page_icon="🏷️", layout="centered")

@st.cache_resource
def download_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk_data()

# ---------- Load model + vocab (cached so it only loads once) ----------
@st.cache_resource
def load_artifacts():
    word_id = joblib.load("word_id.joblib")
    tag_id = joblib.load("tag_id.joblib")
    max_len = joblib.load("max_len.joblib")
    id_tag = {v: k for k, v in tag_id.items()}
    model = load_model("word2vec_bilstm.keras")
    return model, word_id, id_tag, max_len


def predict_tags(sentence, model, word_id, id_tag, max_len):
    words = word_tokenize(sentence)
    if len(words) == 0:
        return []

    encoded = [word_id.get(w, word_id["UNK"]) for w in words]
    encoded = pad_sequences([encoded], maxlen=max_len, padding="post", value=word_id["PAD"])

    prediction = model.predict(encoded, verbose=0)
    prediction = np.argmax(prediction, axis=-1)[0]

    results = []
    for i, word in enumerate(words):
        tag = id_tag.get(prediction[i], "UNK")
        results.append((word, tag))
    return results


# ---------- UI ----------
st.title("🏷️ Part-of-Speech Tagger")
st.write(
    "A BiLSTM model trained on the Penn Treebank corpus, using pretrained "
    "**Word2Vec** embeddings, tags each word in your sentence with its "
    "part of speech."
)

try:
    model, word_id, id_tag, max_len = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Model or vocabulary files not found. Make sure `word2vec_bilstm.keras`, "
        "`word_id.joblib`, `tag_id.joblib`, and `max_len.joblib` are in the same "
        "folder as this app (generate them by running `pos_tagging.py` first)."
    )
    st.stop()

sentence = st.text_input(
    "Enter a sentence:",
    placeholder="e.g. The quick brown fox jumps over the lazy dog"
)

if st.button("Tag PoS", type="primary"):
    if not sentence.strip():
        st.warning("Please enter a sentence first.")
    else:
        with st.spinner("Tagging..."):
            results = predict_tags(sentence, model, word_id, id_tag, max_len)

        if results:
            st.subheader("Results")

            # Table view
            st.table(
                {"Word": [w for w, t in results], "Tag": [t for w, t in results]}
            )


with st.expander("ℹ️ About this model"):
    st.write(
        """
        - **Architecture:** Bidirectional LSTM with a TimeDistributed Dense softmax output layer
        - **Embeddings:** Pretrained Word2Vec (Google News, 300-dim), frozen during training
        - **Training data:** Penn Treebank tagged sentences (NLTK)
        - **Max sequence length:** padded/truncated to a fixed length
        """
    )
