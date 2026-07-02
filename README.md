# Part-of-Speech Tagging with BiLSTM

A deep learning project that tags each word in a sentence with its part of speech (noun, verb, adjective, etc.) using a Bidirectional LSTM trained on the Penn Treebank corpus. Three pretrained embedding types — **Word2Vec**, **GloVe**, and **FastText** — are trained and compared, and the best-performing model is served through an interactive Streamlit app.

## Demo
https://pos-tagging-dhruv.streamlit.app/

## Project Structure

```
.
├── pos_tagging.py       # Trains 3 BiLSTM models (Word2Vec / GloVe / FastText) and compares them
├── predict.py           # CLI script to run predictions with the saved Word2Vec model
├── app.py               # Streamlit web app for interactive tagging
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## How It Works

1. **Data:** Penn Treebank tagged sentences, loaded via NLTK.
2. **Preprocessing:** Words and tags are converted to integer sequences and padded to a fixed length.
3. **Embeddings:** Each model uses a different pretrained embedding matrix (Word2Vec / GloVe / FastText), frozen during training.
4. **Model:** `Embedding → Bidirectional LSTM (128 units, dropout 0.3) → TimeDistributed Dense (softmax)`
5. **Evaluation:** Accuracy, precision, recall, and F1-score are computed and compared across the three embedding types.
6. **Deployment:** The Word2Vec model (best trade-off of speed/accuracy in this setup) is loaded in `app.py` for live predictions.

## Setup & Usage

### 1. Clone and install dependencies

```bash
git clone https://github.com/dhruv-rathi-tech/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

### 2. Train the models

This downloads pretrained embeddings (several GB total) and trains all three models. This step is **required** because the trained model files are not included in this repo (see [Note on model files](#note-on-model-files)).

```bash
python pos_tagging.py
```

This generates:
- `word2vec_bilstm.keras`, `glove_bilstm.keras`, `fasttext_bilstm.keras`
- `word_id.joblib`, `tag_id.joblib`, `max_len.joblib`

### 3. Run predictions from the command line (optional)

```bash
python predict.py
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Note on Model Files

Trained .keras model and .joblib vocabulary files (~19 MB total) are small enough to commit directly to this repo, and are included so the deployed app works out of the box without any extra setup. GitHub's hard limit is 100 MB per file, so this comfortably fits.

If you retrain the models and file sizes grow significantly (e.g. if you add GloVe/FastText model files too), consider Git LFS for anything approaching that limit.


## Results

| Embedding | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| Word2Vec  | 0.977969 | 0.948820 | 0.949411 | 0.948859 |
| GloVe     | 0.969072 | 0.928729 | 0.928980 | 0.928185 |
| FastText  | 0.978225 | 0.948641 | 0.949998 | 0.948424 |


## Tech Stack

- Python, TensorFlow/Keras, Gensim, NLTK
- Streamlit (deployment)
- scikit-learn (evaluation metrics)

## Author

Dhruv Rathi
