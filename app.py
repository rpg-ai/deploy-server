from flask import Flask, request, jsonify
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
import spacy
import re

app = Flask(__name__)

# load the classifier model
pipeline = load("text_classification.joblib")

# NLP Pre process
def nlp_preprocess(text):
    
    lemmas = []

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    for token in doc:
        if not token.is_stop: 
            if (token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ'):
                lemmas.append(token.lemma_)
    
    return clean_text(' '.join(lemmas))

def clean_text(text):
    text = text.lower()
    t = unidecode(text)
    t.encode("ascii")  
    t = re.sub(r'[^a-z]', ' ', t)           #Remove nonalpha
    t = re.sub(r'\s[^a-z]\s', ' ', t)       #Remove nonalpha >> check if is really necessary!?!?
    t = re.sub(r"\b[a-z]{1,2}\b", ' ', t)   #Remove words with 1 or 2 letters
    t = re.sub(' +', ' ', t)                #Remove extra spaces
    t = t.strip()                           #Remove leading and trailing spaces
    return t


@app.route('/', methods=["GET"])
def call_model():
    req = request.args
    predict = pipeline.predict([nlp_preprocess(req.get('action'))])

    result_dict = {
        "player": req.get('player'),
        "action": req.get('action'),
        "prediction": predict[0],#{"Acrobatics": "32.54%"},
        "other_predicts": [{"Athletics": "12.9%"}, {"Survival": "2.41%"}]
    }
    return jsonify(result_dict)


if __name__ == '__main__':
    app.run()
