from unidecode import unidecode
import spacy
import re
from joblib import load
import numpy as np
import os
import wget


def load_model():
    model_name = "text_classification.joblib"
    if model_name not in os.listdir('.'):
        wget.download(os.environ["MODEL_DOWNLOAD_URL"])
        print("To baixando")
    print(f"Saí do download, aqui ja deveria ter: {os.listdir('.')}")
    return load(model_name)


model = load_model()


def nlp_preprocess(text):
    lemmas = []

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    for token in doc:
        if not token.is_stop:
            if token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ':
                lemmas.append(token.lemma_)

    return clean_text(' '.join(lemmas))


def clean_text(text):
    text = text.lower()
    t = unidecode(text)
    t.encode("ascii")
    t = re.sub(r'[^a-z]', ' ', t)  # Remove nonalpha
    t = re.sub(r'\s[^a-z]\s', ' ', t)  # Remove nonalpha >> check if is really necessary!?!?
    t = re.sub(r"\b[a-z]{1,2}\b", ' ', t)  # Remove words with 1 or 2 letters
    t = re.sub(' +', ' ', t)  # Remove extra spaces
    t = t.strip()  # Remove leading and trailing spaces
    return t


def predict(message):
    lemmas = nlp_preprocess(message)
    predict = model.predict([lemmas])
    prob = model.predict_proba([lemmas])

    best_n = np.argsort(prob, axis=1)[:, -3:]
    classes = model.classes_

    prediction = predict[0]
    predictions_list = [{classes[best_n[0, 2]]: f"{round(prob[0, best_n[0, 2]] * 100, 2)}%"},
                        {classes[best_n[0, 1]]: f"{round(prob[0, best_n[0, 1]] * 100, 2)}%"},
                        {classes[best_n[0, 0]]: f"{round(prob[0, best_n[0, 0]] * 100, 2)}%"}]

    return prediction, predictions_list
