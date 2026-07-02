import nltk
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Bidirectional, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping
import gensim.downloader as emb
import joblib

nltk.download("treebank")
from nltk.corpus import treebank

tagged_sentences = treebank.tagged_sents()
print("Number of sentences:", len(tagged_sentences))


sentences = []
tags = []

# split sentences and tags
for sent in tagged_sentences:
    word_seq = []
    tag_seq = []

    for word, tag in sent:
        word_seq.append(word)
        tag_seq.append(tag)

    sentences.append(word_seq)
    tags.append(tag_seq)


# words vocabulary
word_set = set()
for s in sentences:
    for w in s:
        word_set.add(w)

print("Vocabulary size:", len(word_set))

word_id = {
    "PAD": 0,
    "UNK": 1
}
for id, word in enumerate(word_set, start=2):
    word_id[word] = id


# tags vocabulary
tag_set = set()
for t in tags:
    for w in t:
        tag_set.add(w)

tag_id = {
    "PAD": 0
}
for id, tag in enumerate(tag_set, start=1):
    tag_id[tag] = id

print("Number of tags:", len(tag_id))


id_word = {v:k for k,v in word_id.items()}
id_tag = {v:k for k,v in tag_id.items()}

# encoding
encoded_sentences = []
for s in sentences:
    encoded_s = []

    for word in s:
        encoded_s.append(word_id.get(word, word_id["UNK"]))

    encoded_sentences.append(encoded_s)


encoded_tags = []
for t in tags:
    encoded_t = []

    for tag in t:
        encoded_t.append(tag_id[tag])

    encoded_tags.append(encoded_t)


lengths = []

for s in encoded_sentences:
    lengths.append(len(s))

sns.boxplot(x=lengths)
plt.xlabel("No. of words in a sentence")
plt.show()

# padding
X = pad_sequences(encoded_sentences, maxlen=60, padding="post", value=word_id["PAD"])
y = pad_sequences(encoded_tags, maxlen=60, padding="post", value=tag_id["PAD"])


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# model training
def train_model(embedding_name, model_name):
    print(f"\nLoading {embedding_name}...\n")

    embedding = emb.load(embedding_name)

    embedding_dim = 300
    embedding_matrix = np.zeros((len(word_id), embedding_dim))

    for word, idx in word_id.items():
        if word in embedding:
            embedding_matrix[idx] = embedding[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    model = Sequential()
    model.add(Embedding(
            input_dim=len(word_id),
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=60,
            trainable=False
        )
    )
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)))
    model.add(TimeDistributed(Dense(len(tag_id), activation="softmax")))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16, callbacks=[early_stop])
    
    print("\nModel Summary\n")
    model.summary()

    plt.figure(figsize=(6,4))
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"{model_name} accuracy")
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{model_name} loss")
    plt.show()

    
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("\nTest Accuracy:", test_accuracy)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)


    true_tags = []
    pred_tags = []
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] != tag_id["PAD"]:
                true_tags.append(y_test[i][j])
                pred_tags.append(y_pred[i][j])

    print(classification_report(true_tags, pred_tags))
    precision, recall, f1, _ = precision_recall_fscore_support(true_tags, pred_tags, average="weighted")
    
    model.save(model_name + ".keras")
    print(f"\n{model_name} saved successfully.\n")

    return test_accuracy, precision, recall, f1


# models
word2vec_acc, word2vec_pre, word2vec_rec, word2vec_f1 = train_model("word2vec-google-news-300", "word2vec_bilstm")
glove_acc, glove_pre, glove_rec, glove_f1 = train_model("glove-wiki-gigaword-300", "glove_bilstm")
fasttext_acc, fasttext_pre, fasttext_rec, fasttext_f1 = train_model("fasttext-wiki-news-subwords-300", "fasttext_bilstm")


# results
results = pd.DataFrame({
    "Embedding": ["Word2Vec", "GloVe", "FastText"],
    "Accuracy": [word2vec_acc, glove_acc, fasttext_acc],
    "Precision": [word2vec_pre, glove_pre, fasttext_pre],
    "Recall": [word2vec_rec, glove_rec, fasttext_rec],
    "F1 Score": [word2vec_f1, glove_f1, fasttext_f1]
})

print(results)

results.set_index("Embedding").plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Comparison of Embedding Models")
plt.ylabel("Score")
plt.ylim(0.9, 1.0)
plt.grid(axis="y")
plt.legend(loc="lower right")
plt.show()


metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

for metric in metrics:
    plt.figure(figsize=(6,4))
    plt.bar(results["Embedding"], results[metric])

    plt.title(f"{metric} Comparison")
    plt.xlabel("Embedding")
    plt.ylabel(metric)

    plt.ylim(0.9, 1.0)
    plt.grid(axis="y")
    plt.show()


joblib.dump(word_id, "word_id.joblib")
joblib.dump(tag_id, "tag_id.joblib")
joblib.dump(60, "max_len.joblib")
