import nltk
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")

word_id = joblib.load("word_id.joblib")
tag_id = joblib.load("tag_id.joblib")
max_len = joblib.load("max_len.joblib")

id_tag = {v: k for k, v in tag_id.items()}

model = load_model("word2vec_bilstm.keras")


def predict(sentence):
    words = word_tokenize(sentence)
    encoded = []

    for word in words:
        encoded.append(word_id.get(word, word_id["UNK"]))

    encoded = pad_sequences([encoded], maxlen=max_len, padding="post", value=word_id["PAD"])

    prediction = model.predict(encoded, verbose=0)
    prediction = np.argmax(prediction, axis=-1)[0]

    for i in range(len(words)):
        word = words[i]
        tag = id_tag[prediction[i]]

        print(word, ":", tag)


while True:
    sentence = input("\nEnter a sentence (type 'exit' to quit): ")
    if sentence.lower() == "exit":
        break

    predict(sentence)